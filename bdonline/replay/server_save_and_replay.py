#!/usr/bin/env python

'''
Server application that allows for recreation of experiments. The server is hardwired to take the recordings of a
specific experiment and to replay them. The trainer that is used should be fed with the indexes of the supercrops
saved during the training. The results of the simulation will be stored in the file LOG_FILENAME
This application should be run in parallel with the LSL-Adapter replay and the index sender
'''

#
# imports
#
import sys
sys.path.append('D:\\DLVR\\xdf_mne_interface') #path to the xdf_mne_interface package
sys.path.append('D:\\DLVR\\braindecode') #path to the braindecode package
sys.path.append('D:\\braindecode-online')  #path to the braindecode-online package

import os
import os.path
import signal
import argparse
import datetime
import time
from glob import glob
import threading
import logging
from bdonline.parsers import parse_command_line_arguments

import matplotlib

matplotlib_backend = parse_command_line_arguments().plotbackend
try:
    matplotlib.use(matplotlib_backend)
except:
    print("Could not use {:s} backend for matplotlib".format(
        matplotlib_backend))
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

import numpy as np
import torch as th
import pyxdf as xdf
import xdf_to_bd
from scipy.signal import filtfilt, iirnotch, butter
from torch.optim import Adam
from gevent import socket
import gevent.select
import gevent.server
from scipy import interpolate
import h5py

from braindecode.models import deep4
from braindecode.torch_ext.constraints import MaxNormDefaultConstraint
from braindecode.torch_ext.losses import log_categorical_crossentropy
from braindecode.models.util import to_dense_prediction_model

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bdonline.datasuppliers import ArraySupplier
from bdonline.experiment import OnlineExperiment
from bdonline.buffer import DataMarkerBuffer
from bdonline.predictors import DummyPredictor, ModelPredictor
from bdonline.processors import StandardizeProcessor
from bdonline.trainers import NoTrainer
from bdonline.replay.trainers_replay import BatchCntTrainer
from braindevel_online.live_plot import LivePlot
from braindecode.models.util import to_dense_prediction_model
from braindecode.models import deep4
from bdonline.read import read_until_bytes_received, AsyncStdinReader, my_async_stdin_reader

TCP_SENDER_EEG_NCHANS = 65  # number of channels to send to braindecode-online, includes labels
TCP_SENDER_EEG_NSAMPLES = 10



xdf_filenames=['D:\\DLVRData\\HeBoEMGVR3\\data_1.xdf'] #the data we want to replay
DATA_AND_LABELS, CHAN_NAMES = xdf_to_bd.bd_data_from_xdf(xdf_filenames, 250) #Load in all data and labels


class AsyncStdinReader(threading.Thread):
    
    # override for threading.Thread
    def __init__(self):
        self.active = False
        self.input_string = None
        super(AsyncStdinReader, self).__init__()
    
    # override for threading.Thread
    def run(self):
        print('AsyncStdinReader: reading until enter:')
        self.input_string = input()
        self.active = False
    
    def input_async(self):
        if self.active:
            return None
        elif self.input_string is None:
            self.active = True
            self.start()
            return None
        else:
            returnstring = self.input_string
            self.input_string = None
            return returnstring

LOG_FILENAME = 'stored_experiment.txt'
logging.basicConfig(filename=LOG_FILENAME, filemode='w', level=logging.INFO)
log = logging.getLogger(__name__)

TCP_SENDER_EEG_NCHANS = 65  # number of channels to send to braindecode-online, includes labels
TCP_SENDER_EEG_NSAMPLES = 10
CHAN_NAMES = ['Fp1', 'Fpz', 'Fp2', 'AF7',     # channel names send to braindecode-online
                  'AF3', 'AF4', 'AF8', 'F7',
                  'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5',
                  'FC3',
                  'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'M1', 'T7', 'C5',
                  'C3',
                  'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'M2', 'TP7', 'CP5', 'CP3',
                  'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3',
                  'P1',
                  'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POz',
                  'PO4',
                  'PO6', 'PO8', 'O1', 'Oz', 'O2', 'marker']


path = 'D:\\DLVR\\Data\\subjM\\' #the data we want to replay
files = ['block4.xdf']
DATA_AND_LABELS  = xdf_to_bd.dlvr_braindecode(path,files, -1, 250) #Load in all data and labels

class PredictionSender(object):
    def __init__(self, socket):
        self.socket = socket

    def send_prediction(self, i_sample, prediction, label):
        log.info("Prediction for sample {:d}:\n{:s} with label{:s}".format(
            i_sample, str(prediction), str(label)))
        # +1 to convert 0-based to 1-based indexing
        self.socket.sendall("{:d}\n".format(i_sample + 1).encode('utf-8'))
        n_preds = len(prediction) # e.g. number of classes
        # format all preds as floats with spaces inbetween
        format_str = " ".join(["{:f}"] * n_preds) + "\n"
        pred_str = format_str.format(*prediction)
        self.socket.sendall(pred_str.encode('utf-8'))
        label_str = str(label) + "\n"
        self.socket.sendall(label_str.encode('utf-8'))


class DataFromFile():
    """ mimick the receiving of data through LSL, it pulls indexes from a stream
        which is created by the index sender
    """
    def __init__(self, socket):
        self.socket = socket

    def wait_for_data(self):
        idx = self.socket.recv(128)
        idx = np.frombuffer(idx, dtype='int64')
        try:
            array = DATA_AND_LABELS[:, idx]
            data = array[:-1, :].T
            markers = array[-1, :]
            return data, markers

        except IndexError:
            return None



class PredictionServer(gevent.server.StreamServer):
    def __init__(self, listener, online_experiment, out_hostname, out_port,
        plot_sensors, use_out_server, save_data,
                 model_base_name,
                 save_model_trainer_params,
            handle=None, backlog=None, spawn='default', **ssl_args):
        """
        adapt_model only needed to know for saving
        """
        self.online_experiment = online_experiment
        self.out_hostname = out_hostname
        self.out_port = out_port
        self.plot_sensors = plot_sensors
        self.use_out_server = use_out_server
        self.save_data = save_data
        self.model_base_name = model_base_name
        self.save_model_trainer_params = save_model_trainer_params
        super(PredictionServer, self).__init__(listener, handle=handle, spawn=spawn)

    def handle(self, in_socket, address):
        log.info('New connection from {:s}!'.format(str(address)))
        # Connect to out Server
        if self.use_out_server:
            gevent.sleep(1) # hack to wait until out server open
            prediction_sender = self.connect_to_prediction_receiver()
            log.info("Connected to out Server")
        else:
            prediction_sender=None

        # Receive Header
        chan_names, n_chans, n_samples_per_block = CHAN_NAMES, TCP_SENDER_EEG_NCHANS, TCP_SENDER_EEG_NSAMPLES
        n_numbers = n_chans * n_samples_per_block
        n_bytes_per_block = n_numbers * 4 # float32
        data_receiver = DataFromFile(in_socket)

        log.info("Numbers in total:  {:d}".format(n_numbers))
        
        log.info("Before checking plot")
        # Possibly plot
        if self.plot_sensors:
            self.plot_sensors_until_enter_press(chan_names, in_socket,
                                                n_bytes_per_block,
            n_chans, n_samples_per_block)
        log.info("After checking plot")
        self.make_predictions_until_enter_press(chan_names, n_chans,
                                              n_samples_per_block,
            n_bytes_per_block, data_receiver, prediction_sender)
        now = datetime.datetime.now()
        if self.save_data:
            self.save_signal_markers(now)
        if self.save_model_trainer_params:
            self.save_params(now)
        self.print_results()
        self.stop()

    def connect_to_prediction_receiver(self):
        out_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("Out server connected at:", self.out_hostname, self.out_port)
        out_socket.connect((self.out_hostname, self.out_port))
        return PredictionSender(out_socket)

    def receive_header(self, in_socket):
        chan_names_line = '' + in_socket.recv(1).decode('utf-8')
        while len(chan_names_line) == 0 or chan_names_line[-1] != '\n':
            chan_names_line += in_socket.recv(1).decode('utf-8')
        log.info("Chan names:\n{:s}".format(chan_names_line))
        chan_names = chan_names_line.replace('\n','').split(" ")
        
        #
        assert np.array_equal(chan_names, ['Fp1', 'Fpz', 'Fp2', 'AF7',
         'AF3', 'AF4', 'AF8', 'F7',
         'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3',
         'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'M1', 'T7', 'C5', 'C3',
         'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'M2', 'TP7', 'CP5', 'CP3',
         'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1',
         'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POz', 'PO4',
         'PO6', 'PO8', 'O1', 'Oz', 'O2', 'marker']
            ) or np.array_equal(chan_names,
         ['Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8',
         'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'FC1', 'FCz',
         'FC2', 'C3', 'C1', 'Cz', 'C2', 'C4', 'CP3', 'CP1', 'CPz',
         'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz', 'marker'])
        n_rows = read_until_bytes_received(in_socket, 4)#n_chans+marker
        n_rows = np.fromstring(n_rows, dtype=np.int32)[0]
        log.info("Number of rows:    {:d}".format(n_rows))
        n_cols = read_until_bytes_received(in_socket, 4)#n_samples_per_block
        n_cols = np.fromstring(n_cols, dtype=np.int32)[0]
        log.info("Number of columns: {:d}".format(n_cols))
        assert n_rows == len(chan_names)
        return chan_names, n_rows, n_cols


    def plot_sensors_until_enter_press(self, chan_names, in_socket, n_bytes,
                                       n_chans, n_samples_per_block):
        log.info("Starting Plot for plot")
        live_plot = LivePlot(plot_freq=150)
        log.info("Liveplot created")
        live_plot.initPlots(chan_names)
        log.info("Initialized")
        enter_pressed = False
        while not enter_pressed:
            array = ''
            while len(array) < n_bytes:
                array += in_socket.recv(n_bytes - len(array))
            array = np.fromstring(array, dtype=np.float32)
            array = array.reshape(n_chans, n_samples_per_block, order='F')
            live_plot.accept_samples(array.T)
            # check if enter is pressed
            # throws exception on Windows. 
            # i,o,e = gevent.select.select([sys.stdin],[],[],0.001)
            # for s in i:
            #     if s == sys.stdin:
            #         _ = sys.stdin.readline()
            #         enter_pressed = True
            input_string = my_async_stdin_reader.input_async()
            if input_string is not None:
                enter_pressed = True

        live_plot.close()
        log.info("Plot finished")


    def make_predictions_until_enter_press(self, chan_names, n_chans,
                                         n_samples_per_block, n_bytes,
                                         data_receiver, prediction_sender):
        

        self.online_experiment.set_supplier(data_receiver)
        # -1 because n_chans includes marker chan
        self.online_experiment.set_buffer(DataMarkerBuffer(n_chans - 1, 20000))
        self.online_experiment.set_sender(prediction_sender)
        self.online_experiment.trainer.set_n_chans(n_chans - 1)
        self.online_experiment.run()
        print("Finished online experiment")
        return
            

    def print_results(self):
        n_classes = 2
        i_samples = list(zip(
            *self.online_experiment.storer.i_samples_and_predictions))[0]
        predictions = list(zip(
            *self.online_experiment.storer.i_samples_and_predictions))[1]
        predictions = np.array(predictions)
        markers = np.concatenate(self.online_experiment.storer.marker_blocks)

        interpolate_fn = interpolate.interp1d(i_samples, predictions.T,
                                              bounds_error=False, fill_value=0)
        interpolated_preds = interpolate_fn(range(0, len(markers)))
        markers_1_hot = np.zeros((len(markers), n_classes))
        for i_class in range(n_classes):
            markers_1_hot[:, i_class] = (markers == (i_class + 1))

        markers_1_hot[:, -1] = ((markers == n_classes) | (markers == -1))
        corrcoeffs = np.corrcoef(interpolated_preds, markers_1_hot.T)[
                     n_classes:,:n_classes]

        print("Corrcoeffs (assuming n_classes=", n_classes, ")")
        print(corrcoeffs)
        print("diagonal")
        print(np.diag(corrcoeffs))
        print("mean across diagonal")
        print(np.mean(np.diag(corrcoeffs)))


        # inside trials
        corrcoeffs = np.corrcoef(interpolated_preds[:,markers!=0],
                                 markers_1_hot[markers!=0].T)[:n_classes,n_classes:]
        print("Corrcoeffs inside trial")
        print(corrcoeffs)
        print("diagonal")
        print(np.diag(corrcoeffs))
        print("mean across diagonal inside trial")
        print(np.mean(np.diag(corrcoeffs)))

        # Accuracies
        interpolated_pred_labels = np.argmax(interpolated_preds, axis=0)
        # -1 since we have 0 as "break" "non-trial" marker
        label_pred_equal = interpolated_pred_labels == markers - 1
        label_pred_trial_equal = label_pred_equal[markers != 0]
        print("Sample accuracy inside trials")
        print(np.mean(label_pred_trial_equal))
        markers_with_breaks = np.copy(markers)
        # set break to rest label
        markers_with_breaks[markers_with_breaks == 0] = n_classes
        # from 1-based to 0-based
        markers_with_breaks -= 1
        label_pred_equal = interpolated_pred_labels == markers_with_breaks
        print("Sample accuracy total")
        print(np.mean(label_pred_equal))

        # also compute trial preds
        # compute boundarides so that boundaries give
        # indices of starts of new trials/new breaks
        trial_labels = []
        trial_pred_labels = []
        boundaries = np.flatnonzero(np.diff(markers) != 0) + 1
        last_bound = 0
        for i_bound in boundaries:
            # i bounds are first sample of new trial
            this_labels = markers_with_breaks[last_bound:i_bound]
            assert len(np.unique(this_labels) == 1), (
                "Expect only one label, got {:s}".format(str(
                    np.unique(this_labels))))
            trial_labels.append(this_labels[0])
            this_preds = interpolated_preds[:, last_bound:i_bound]
            pred_label = np.argmax(np.mean(this_preds, axis=1))
            trial_pred_labels.append(pred_label)
            last_bound = i_bound
        trial_labels = np.array(trial_labels)
        trial_pred_labels = np.array(trial_pred_labels)
        print(
        "Trialwise accuracy (mean prediction) of {:d} trials (including breaks, without offset)".format(
            len(trial_labels)))
        print(np.mean(trial_labels == trial_pred_labels))


    def save_signal_markers(self, now):
        now_timestring = now.strftime('%Y-%m-%d_%H-%M-%S')
        all_data = np.concatenate(self.online_experiment.storer.data_blocks)
        all_markers = np.concatenate(
            self.online_experiment.storer.marker_blocks)
        combined = np.concatenate((all_data, all_markers[:, None]),
                                  axis=1).astype(np.float32)
        day_string = now.strftime('%Y-%m-%d')
        data_folder = 'data/online/{:s}'.format(day_string)
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
        data_filename = os.path.join(data_folder, "{:s}.npy".format(
                now_timestring))
        log.info("Saving data to {:s}".format(data_filename))
        np.save(data_filename, combined)


    def save_params(self, now):
        now_timestring = now.strftime('%Y-%m-%d_%H-%M-%S')
        filename = os.path.join(
            self.model_base_name,
            "{:s}.model_params.pkl".format(now_timestring))
        log.info("Save model params to {:s}".format(filename))
        th.save(self.online_experiment.trainer.model.state_dict(), filename)
        filename = os.path.join(
            self.model_base_name,
            "{:s}.trainer_params.pkl".format(now_timestring))
        log.info("Save trainer params to {:s}".format(filename))
        th.save(self.online_experiment.trainer.optimizer.state_dict(), filename)


def get_now_timestring():
    now = datetime.datetime.now()
    time_string = now.strftime('%Y-%m-%d_%H-%M-%S')
    return time_string      


def setup_logging():
    """ Set up a root logger so that other modules can use logging
    Adapted from scripts/train.py from pylearn"""

    logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                        level=logging.DEBUG, stream=sys.stdout)


def main(
        out_hostname, out_port, base_name, params_filename, plot_sensors,
        use_out_server, adapt_model, save_data, n_updates_per_break, batch_size,
        learning_rate, n_min_trials, trial_start_offset, break_start_offset,
        break_stop_offset,
        pred_gap,
        incoming_port, load_old_data, use_new_adam_params,
        input_time_length,
        train_on_breaks,
        min_break_samples,
        min_trial_samples,
        n_chans,
        cuda,
        savegrad, gradfolder):
    setup_logging()

    hostname = 'localhost'
    supplier = None
    sender = None
    buffer = None
    # load model to correct gpu id
    model_name = os.path.join(base_name, 'deep_4_params')
    model_dict = th.load(model_name)
    final_conv_length = 2
    model = deep4.Deep4Net(n_chans, 2, input_time_length, final_conv_length)
    model = model.create_network()
    model.load_state_dict(model_dict)
    to_dense_prediction_model(model)
    model.cuda()

    predictor = ModelPredictor(
        model, input_time_length=input_time_length, pred_gap=pred_gap,
        cuda=cuda)
    if adapt_model:
        #loss_function = log_categorical_crossentropy
        loss_function = th.nn.CrossEntropyLoss()

        model_loss_function = None
        model_constraint = MaxNormDefaultConstraint()
        optimizer = Adam(model.parameters(), lr=learning_rate)
        n_preds_per_input = None # set later
        n_classes = None # set later
        trainer = BatchCntTrainer(
            model, loss_function, model_loss_function, model_constraint,
            optimizer, input_time_length,
            n_preds_per_input, n_classes,
            n_updates_per_break=n_updates_per_break,
            batch_size=batch_size,
            n_min_trials=n_min_trials,
            trial_start_offset=trial_start_offset,
            break_start_offset=break_start_offset,
            break_stop_offset=break_stop_offset,
            add_breaks=train_on_breaks,
            min_break_samples=min_break_samples,
            min_trial_samples=min_trial_samples,
            cuda=cuda,
            savegrad=savegrad,
            gradfolder=gradfolder)
        trainer.set_n_chans(n_chans)
    else:
        trainer = NoTrainer()

    if params_filename is not None:
        if params_filename == 'newest':
            # sort will already sort temporally with our time string format
            all_params_files = sorted(glob(os.path.join(base_name,
                                                        "*.model_params.pkl")))
            assert len(all_params_files) > 0, ("Expect atleast one params file "
                "if 'newest' given as argument")
            params_filename = all_params_files[-1]
        log.info("Loading model params from {:s}".format(params_filename))
        model_params = th.load(params_filename, map_location=inner_device_mapping)
        model.load_state_dict(model_params)
        train_params_filename = params_filename.replace('model_params.pkl',
                                                        'trainer_params.pkl')
        if os.path.isfile(train_params_filename):
            if adapt_model and use_new_adam_params:
                log.info("Loading trainer params from {:s}".format(
                    train_params_filename))
                train_params = th.load(train_params_filename,
                                       map_location=inner_device_mapping)
                optimizer.load_state_dict(train_params)
        elif adapt_model:
            log.warn("No train/adam params found, starting optimization params "
                     "from scratch (model params will be loaded anyways).")
    processor = StandardizeProcessor()
    if adapt_model and load_old_data:
        trainer.add_data_from_today(
            factor_new=processor.factor_new, eps=processor.eps)
    online_exp = OnlineExperiment(
        supplier=supplier, buffer=buffer,
        processor=processor,
        predictor=predictor, trainer=trainer, sender=sender)
    server = PredictionServer(
        (hostname, incoming_port),
        online_experiment=online_exp,
        out_hostname=out_hostname, out_port=out_port,
        plot_sensors=plot_sensors,
        use_out_server=use_out_server, save_data=save_data,
        model_base_name=base_name,
        save_model_trainer_params=adapt_model)
    # Compilation takes some time so initialize trainer already
    # before waiting in connection in server
    log.info("Starting server on port {:d}".format(incoming_port))
    server.start()
    log.info("Started server")
    print('started server')

    server.serve_forever()

if __name__ == '__main__':
    gevent.signal(signal.SIGINT, gevent.kill)
    args = parse_command_line_arguments()
    if args.noprint:
        log.setLevel("WARN")
    # factor for converting to samples
    ms_to_samples = args.fs / 1000.0
    # convert all millisecond arguments to number of samples
    main(
        out_hostname=args.outhost,
        out_port=args.outport,
        base_name=args.expfolder,
        params_filename=args.paramsfile,
        plot_sensors=args.plot,
        save_data=not args.nosave,
        use_out_server=not args.noout,
        adapt_model=not args.noadapt,
        n_updates_per_break=args.updatesperbreak,
        batch_size=args.batchsize,
        learning_rate=args.learningrate,
        n_min_trials=args.mintrials, 
        trial_start_offset=int(args.trialstartoffsetms * ms_to_samples),
        break_start_offset=int(args.breakstartoffsetms * ms_to_samples),
        break_stop_offset=int(args.breakstopoffsetms * ms_to_samples),
        pred_gap=int(args.predgap * ms_to_samples),
        incoming_port=args.inport,
        load_old_data=not args.noolddata,
        use_new_adam_params=not args.nooldadamparams,
        input_time_length=args.inputsamples,
        train_on_breaks=(not args.nobreaktraining),
        min_break_samples=int(args.minbreakms * ms_to_samples),
        min_trial_samples=int(args.mintrialms * ms_to_samples),
        n_chans=args.nchans,
        cuda=not args.cpu,
        savegrad=args.savegrad,
        gradfolder=args.gradfolder)
    
