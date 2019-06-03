from gevent import socket
import signal
import numpy as np
from numpy.random import RandomState
import gevent.server
import gevent.select
import sys
from scipy import interpolate
import logging

log = logging.getLogger(__name__)


class RememberPredictionsServer(gevent.server.StreamServer):
    def __init__(self, listener,
                 handle=None, backlog=None, spawn='default', **ssl_args):
        super(RememberPredictionsServer, self).__init__(listener,
                                                        handle=handle,
                                                        spawn=spawn)
        self.all_preds = []
        self.i_pred_samples = []

    def handle(self, socket, address):
        print ("new connection")
        # using a makefile because we want to use readline()
        socket_file = socket.makefile()
        while True:
            i_sample = socket_file.readline()
            preds = socket_file.readline()
            print("Number of predictions", len(self.i_pred_samples) + 1)
            print(i_sample[:-1])  # :-1 => without newline
            print(preds[:-1])
            self.all_preds.append(preds)
            self.i_pred_samples.append(i_sample)
            print("")


def start_remember_predictions_server():
    hostname = 'localhost'
    server = RememberPredictionsServer((hostname, 30000))
    print("Starting server")
    server.start()
    print("Started server")
    return server


def send_file_data():

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("127.0.0.1", 7987))
    gevent.sleep(2) # allow other server some time to react to connection first

    chan_names = ['Fp1', 'Fpz', 'Fp2', 'AF7',
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

    chan_line = " ".join(chan_names) + "\n"
    s.send(chan_line.encode('utf-8'))
    n_chans = 65
    n_samples = 100
    s.send(np.array([n_chans], dtype=np.int32).tobytes())
    s.send(np.array([n_samples], dtype=np.int32).tobytes())
    print("Sending data...")
    i_block = 0  # if setting i_block to sth higher, printed results will incorrect
    max_stop_block = 10000
    stop_block = 1000
    cur_marker = 0
    n_samples_waiting = 49
    n_to_next_marker = n_samples_waiting
    rng = RandomState(3948394)
    assert stop_block < max_stop_block
    while i_block < stop_block:
        # chan x time
        arr = rng.randn(n_chans, n_samples).astype(np.float32)

        arr[-1,:] = cur_marker
        s.send(arr.tobytes(order='F'))
        assert arr.shape == (n_chans, n_samples)
        i_block += 1
        gevent.sleep(0.03)
        n_to_next_marker -= n_samples
        if n_to_next_marker <= 0:
            n_to_next_marker = n_samples_waiting
            if cur_marker  == 0:
                cur_marker = float(rng.randint(1,6))
            else:
                cur_marker = 0
    print("Done.")


def create_y_labels(cnt):
    classes = np.unique([m[1] for m in cnt.markers])
    if np.array_equal(range(1, 5), classes):
        return create_y_labels_fixed_trial_len(cnt, trial_len=int(cnt.fs * 4))
    elif np.array_equal(range(1, 9), classes):
        y_signal = create_cnt_y_start_end_marker(cnt,
                                                 start_marker_def=dict((('1',
                                                                         [1]), (
                                                                        '2',
                                                                        [2]), (
                                                                        '3',
                                                                        [3]), (
                                                                        '4',
                                                                        [4]))),
                                                 end_marker_def=dict((
                                                                     ('1', [5]),
                                                                     ('2', [6]),
                                                                     ('3', [7]),
                                                                     ('4',
                                                                      [8]))),
                                                 segment_ival=(0, 0),
                                                 timeaxis=-2)
        y_labels = np.zeros((cnt.data.shape[0]), dtype=np.int32)
        y_labels[y_signal[:, 0] == 1] = 1
        y_labels[y_signal[:, 1] == 1] = 2
        y_labels[y_signal[:, 2] == 1] = 3
        y_labels[y_signal[:, 3] == 1] = 4
        return y_labels
    else:
        raise ValueError("Expect classes 1,2,3,4, possibly with end markers "
                         "5,6,7,8, instead got {:s}".format(str(classes)))


def has_fixed_trial_len(cnt):
    classes = np.unique([m[1] for m in cnt.markers])
    if np.array_equal(range(1, 5), classes):
        return True
    elif np.array_equal(range(1, 9), classes):
        return False
    else:
        raise ValueError("Expect classes 1,2,3,4, possibly with end markers "
                         "5,6,7,8, instead got {:s}".format(str(classes)))


def create_y_labels_fixed_trial_len(cnt, trial_len):
    fs = cnt.fs
    event_samples_and_classes = [(int(np.round(m[0] * fs / 1000.0)), m[1])
                                 for m in cnt.markers]
    y = np.zeros((cnt.data.shape[0]), dtype=np.int32)
    for i_sample, marker in event_samples_and_classes:
        assert marker in [1, 2, 3, 4], "Assuming 4 classes for now..."
        y[i_sample:i_sample + trial_len] = marker
    return y


def create_y_signal(cnt):
    if has_fixed_trial_len(cnt):
        return create_y_signal_fixed_trial_len(cnt, trial_len=int(cnt.fs * 4))
    else:
        return create_cnt_y_start_end_marker(cnt,
                                             start_marker_def=dict((('1', [1]),
                                                                    ('2', [2]),
                                                                    ('3', [3]),
                                                                    (
                                                                    '4', [4]))),
                                             end_marker_def=dict((('1', [5]),
                                                                  ('2', [6]),
                                                                  ('3', [7]),
                                                                  ('4', [8]))),
                                             segment_ival=(0, 0), timeaxis=-2)


def create_y_signal_fixed_trial_len(cnt, trial_len):
    fs = cnt.fs
    event_samples_and_classes = [(int(np.round(m[0] * fs / 1000.0)), m[1]) for m
                                 in cnt.markers]
    return get_y_signal(cnt.data, event_samples_and_classes, trial_len)


def get_y_signal(cnt_data, event_samples_and_classes, trial_len):
    # Generate class "signals", rest always inbetween trials
    y = np.zeros((cnt_data.shape[0], 4), dtype=np.int32)
    y[:, 2] = 1  # put rest class everywhere
    for i_sample, marker in event_samples_and_classes:
        i_class = marker - 1
        # set rest to zero, correct class to 1
        y[i_sample:i_sample + trial_len, 2] = 0
        y[i_sample:i_sample + trial_len, i_class] = 1
    return y


if __name__ == "__main__":
    # load file as cnt
    # send sensor NAMES
    # + MARKER as final name

    # number of rows + number of columns
    # send all as fast as can be
    # also start server, should also expect a quit signal....
    # (for now just stop reading after 10 preds)
    # wait for enter press to continue
    gevent.signal(signal.SIGINT, gevent.kill)
    server = start_remember_predictions_server()
    send_file_data()

    
    print("Finished sending data, press enter to continue")
    
    input()
    
    """
    enter_pressed = False
    while not enter_pressed:
        i, o, e = gevent.select.select([sys.stdin], [], [], 0.1)
        for s in i:
            if s == sys.stdin:
                _ = sys.stdin.readline()
                enter_pressed = True
    """            
                
    """Not needed anymore? reactivate this?
    y_signal = create_y_signal(cnt)
    i_pred_samples = [int(line[:-1]) for line in server.i_pred_samples]
    # -1 to convert from 1 to 0-based indexing
    i_pred_samples_arr = np.array(i_pred_samples) - 1
    preds = [[float(num_str) for num_str in line_str[:-1].split(' ')] 
        for line_str in server.all_preds]
    preds_arr = np.array(preds)

    input_start = 499
    input_end = i_pred_samples_arr[-1] + 1
    interpolated_classes = y_signal[input_start:input_end].T
    interpolate_fn = interpolate.interp1d(i_pred_samples_arr, preds_arr.T,
                                         bounds_error=False, fill_value=0)
    interpolated_preds = interpolate_fn(range(input_start,input_end))
    corrcoeffs = np.corrcoef(interpolated_preds, interpolated_classes)[:4,4:]
    print corrcoeffs"""
