#!/usr/bin/env python

def parse_command_line_arguments():
    import argparse
    parser = argparse.ArgumentParser(
        description="""Launch server for online decoding.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # see http://stackoverflow.com/a/24181138/1469195
    required_named = parser.add_argument_group('required named arguments')
    required_named.add_argument('--fs', action='store', type=int,
        help="Sampling rate of EEG signal (in Hz). Only used to convert "
        "other arguments from milliseconds to number of samples", required=True)
    required_named.add_argument('--expfolder', action='store',
        help='Folder with model etc.', required=True)
    required_named.add_argument('--inputsamples', action='store',
        type=int,
        help='Input samples (!) for the ConvNet (in samples!).', required=True)
    parser.add_argument('--inport', action='store', type=int,
        default=7987, help='Port from which to accept incoming sensor data.')
    parser.add_argument('--outhost', action='store',
        default='172.30.0.117', help='Hostname/IP of the prediction receiver')
    parser.add_argument('--outport', action='store', type=int,
        default=30000, help='Port of the prediction receiver')
    parser.add_argument('--paramsfile', action='store', 
        help='Use these (possibly adapted) parameters for the model. '
        'Filename should end with model_params.npy. Can also use "newest"'
        'to load the newest available  parameter file. '
        'None means to not load any new parameters, instead use '
        'originally (offline)-trained parameters.')
    parser.add_argument('--plot', action='store_true',
        help="Show plots of the sensors first.")
    parser.add_argument('--noout', action='store_true',
        help="Don't wait for prediction receiver.")
    parser.add_argument('--noadapt', action='store_true',
        help="Don't adapt model while running online.")
    parser.add_argument('--updatesperbreak', action='store', default=5,
        type=int, help="How many updates to adapt the model during trial break.")
    parser.add_argument('--batchsize', action='store', default=45, type=int,
        help="Batch size for adaptation updates.")
    parser.add_argument('--learningrate', action='store', default=1e-4, 
        type=float, help="Learning rate for adaptation updates.")
    parser.add_argument('--mintrials', action='store', default=10, type=int,
        help="Number of trials before starting adaptation updates.")
    parser.add_argument('--trialstartoffsetms', action='store', default=500, type=int,
        help="Time offset for the first sample to use (within a trial, in ms) "
        "for adaptation updates.")
    parser.add_argument('--breakstartoffsetms', action='store', default=1000, type=int,
        help="Time offset for the first sample to use (within a break(!), in ms) "
        "for adaptation updates.")
    parser.add_argument('--breakstopoffsetms', action='store', default=-1000, type=int,
        help="Sample offset for the last sample to use (within a break(!), in ms) "
        "for adaptation updates.")
    parser.add_argument('--predgap', action='store', default=200, type=int,
        help="Amount of milliseconds between predictions.")
    parser.add_argument('--minbreakms', action='store', default=2000, type=int,
        help="Minimum length of a break to be used for training (in ms).")
    parser.add_argument('--mintrialms', action='store', default=0, type=int,
        help="Minimum length of a trial to be used for training (in ms).")
    parser.add_argument('--noprint', action='store_true',
        help="Don't print on terminal.")
    parser.add_argument('--nosave', action='store_true',
        help="Don't save streamed data (including markers).")
    parser.add_argument('--noolddata', action='store_true',
        help="Dont load and use old data for adaptation")
    parser.add_argument('--plotbackend', action='store',
        default='agg', help='Matplotlib backend to use for plotting.')
    parser.add_argument('--nooldadamparams', action='store_true',
        help='Do not load old adam params.')
    parser.add_argument('--nobreaktraining',action='store_true',
        help='Do not use the breaks as training examples for the rest class.')
    parser.add_argument('--cpu',action='store_true',
        help='Use the CPU instead of GPU/Cuda.')
    parser.add_argument('--nchans', action='store', default=64, type=int,
        help="Number of EEG channels")
    args = parser.parse_args()
    assert args.breakstopoffsetms <= 0, ("Please supply a nonpositive break stop "
        "offset, you supplied {:d}".format(args.breakstopoffset))
    return args

    
#
# imports
#
import sys
import os
import os.path
import signal
import argparse
import datetime
import time
from glob import glob
import threading
import logging

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
from torch.optim import Adam
from gevent import socket
import gevent.select
import gevent.server
from scipy import interpolate
import h5py

from braindecode.torch_ext.constraints import MaxNormDefaultConstraint
from braindecode.torch_ext.losses import log_categorical_crossentropy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bdonline.datasuppliers import ArraySupplier
from bdonline.experiment import OnlineExperiment
from bdonline.buffer import DataMarkerBuffer
from bdonline.predictors import DummyPredictor, ModelPredictor
from bdonline.processors import StandardizeProcessor
from bdonline.trainers import NoTrainer
from bdonline.trainers import BatchCntTrainer
from braindevel_online.live_plot import LivePlot



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

            
#
# global variables
#
log = logging.getLogger(__name__)
my_async_stdin_reader = AsyncStdinReader()            
            
            


class PredictionSender(object):
    def __init__(self, socket):
        self.socket = socket

    def send_prediction(self, i_sample, prediction):
        log.info("Prediction for sample {:d}:\n{:s}".format(
            i_sample, str(prediction)))
        # +1 to convert 0-based to 1-based indexing
        self.socket.sendall("{:d}\n".format(i_sample + 1).encode('utf-8'))
        n_preds = len(prediction) # e.g. number of classes
        # format all preds as floats with spaces inbetween
        format_str = " ".join(["{:f}"] * n_preds) + "\n"
        pred_str = format_str.format(*prediction)
        self.socket.sendall(pred_str.encode('utf-8'))


class DataReceiver(object):
    def __init__(self, socket, n_chans, n_samples_per_block, n_bytes_per_block):
        self.socket = socket
        self.n_chans = n_chans
        self.n_samples_per_block = n_samples_per_block
        self.n_bytes_per_block = n_bytes_per_block

    def wait_for_data(self):
        array = read_until_bytes_received_or_enter_pressed(
            self.socket, self.n_bytes_per_block)
        if array is None:
            return None
        else:
            array = np.fromstring(array, dtype=np.float32)
            array = array.reshape(self.n_chans, self.n_samples_per_block,
                                  order='F')
            data = array[:-1].T
            markers = array[-1]
            return data, markers


def read_until_bytes_received(socket, n_bytes):
    array_parts = []
    n_remaining = n_bytes
    while n_remaining > 0:
        chunk = socket.recv(n_remaining)
        array_parts.append(chunk)
        n_remaining -= len(chunk)
    array = b"".join(array_parts)
    return array
        
    
    
def read_until_bytes_received_or_enter_pressed(socket, n_bytes):
    '''
    Read bytes from socket until reaching given number of bytes, cancel
    if enter was pressed.

    Parameters
    ----------
    socket:
        Socket to read from.
    n_bytes: int
        Number of bytes to read.
    '''
    enter_pressed = False
    # http://dabeaz.blogspot.de/2010/01/few-useful-bytearray-tricks.html
    array_parts = []
    n_remaining = n_bytes
    while (n_remaining > 0) and (not enter_pressed):
        chunk = socket.recv(n_remaining)
        array_parts.append(chunk)
        n_remaining -= len(chunk)
        # check if enter is pressed
        # throws exception on windows. needed?->yes! when stopped the program saves model and data
        # i, o, e = gevent.select.select([sys.stdin], [], [], 0.0001)
        # for s in i:
        #     if s == sys.stdin:
        #         _ = sys.stdin.readline()
        #         enter_pressed = True
        input_string = my_async_stdin_reader.input_async()
        if input_string is not None:
            enter_pressed = True
            
    if enter_pressed:
        return None
    else:
        array = b"".join(array_parts)
        assert len(array) == n_bytes
        return array


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
        chan_names, n_chans, n_samples_per_block = self.receive_header(in_socket)
        n_numbers = n_chans * n_samples_per_block
        n_bytes_per_block = n_numbers * 4 # float32
        data_receiver = DataReceiver(in_socket,  n_chans, n_samples_per_block,
                                     n_bytes_per_block)
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
        cuda):
    setup_logging()

    hostname = 'localhost'
    supplier = None
    sender = None
    buffer = None
    # load model to correct gpu id
    gpu_id = th.FloatTensor(1).cuda().get_device()
    def inner_device_mapping(storage, location):
        return storage.cuda(gpu_id)
    model_name = os.path.join(base_name, 'model.pkl')
    model = th.load(model_name, map_location=inner_device_mapping)

    predictor = ModelPredictor(
        model, input_time_length=input_time_length, pred_gap=pred_gap,
        cuda=cuda)
    if adapt_model:
        loss_function = log_categorical_crossentropy
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
            cuda=cuda)
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
        )
    
