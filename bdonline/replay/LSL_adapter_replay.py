"""
Receive EEG-data from NeurOne and labels from the Unity game both via lsl.
Pass EEG-data and labels to braindecode-online via TCP/IP.
Simultaniously receive predictions from braindecode-online and publish them on lsl.
"""


import sys
import logging
import time
import functools
import numpy as np
from numpy.random import RandomState
from scipy import interpolate
from gevent import socket
import gevent.server
import gevent.select
from socket import timeout as socket_timeout
import signal
import pylsl as lsl
from scipy.signal import filtfilt, iirnotch, butter
from scipy.stats import ttest_ind
from psychopy.clock import wait


# log = logging.getLogger(__name__)
print = functools.partial(print, flush=True)



lsl_inlet_eeg = None                # lsl.StreamInlet to receive eeg from NeurOne
lsl_inlet_labels = None             # lsl.StreamInlet to receive labels from the Unity game
lsl_outlet_predictions = None       # lsl.StreamOutlet to publish predictions
tcp_recv_server_preds = None        # TCP/IP server to receive predictions from braindecode-online
tcp_recv_socket_preds = None        # TCP/IP socket to receive predictions from braindecode-online
tcp_send_socket_eeg = None          # TCP/IP socket to send eeg data to braindecode-online
tcp_recv_preds_connected = False    # indicates if the tcp receiver for predictions is connected


DEBUG = False                                       # DEBUG=True activates extra debug messages

TCP_SENDER_EEG_PORT = 7987                          # port of braindecode-online
TCP_SENDER_EEG_HOSTNAME = '127.0.0.1'               # hostname of braindecode-online
TCP_SENDER_EEG_NCHANS = 65                          # number of channels to send to braindecode-online, includes labels
TCP_SENDER_EEG_NSAMPLES = 400# number of samples * DOWNSAMPLING_COEF per channel to send to braindecode-online at once

LSL_RECEIVER_EEG_NAME = 'NeuroneStream'             # name of the lsl stream to receive EEG data

LSL_RECEIVER_LABELS_NAME = 'Game State'             # name of the lsl stram to receive labels from unity game

LSL_SENDER_PREDS_NAME = 'bdonline'                  # name of the lsl stream to publish predictions
LSL_SENDER_PREDS_TYPE = 'EEG'                       # type of the lsl stream to publish predictions
LSL_SENDER_PREDS_STREAMCHANNELS = 3                 # channels of the lsl stream to publish preds.
LSL_SENDER_PREDS_ID = 'braindecode preds'           # id of the lsl stream to publish preds.

TCP_RECEIVER_PREDS_HOSTNAME = 'localhost'           # hostname of this PC, used to receive predictions
TCP_RECEIVER_PREDS_PORT = 30000                     # port on this PC, used to receive predictions


PREDICTION_NUM_CLASSES = 2
TARGET_FS = 250
DOWNSAMPLING_COEF = int(5000 / TARGET_FS)
PRED_WINDOW_SIZE = 10
PRED_THRESHOLD = 0.6
ACTION_THRESHOLD = 0.8
# Butter filter (highpass) for 1 Hz
B_1, A_1 = butter(6, 1 / TARGET_FS, btype='high', output='ba')

# Butter filter (lowpass) for 30 Hz
B_40, A_40 = butter(6, 40 / TARGET_FS, btype='low', output='ba')

# Notch filter with 50 HZ
F0 = 50.0
Q = 30.0  # Quality factor
# Design notch filter
B_50, A_50 = iirnotch(F0, Q, TARGET_FS)

                  
class PredictionReceiveServer(gevent.server.StreamServer):
    def __init__(self, listener,
                 handle=None, backlog=None, spawn='default', **ssl_args):
        super(PredictionReceiveServer, self).__init__(listener,
                                                        handle=handle,
                                                        spawn=spawn)
        self.all_preds = []
        self.i_pred_samples = []
        self.connected = False

    def handle(self, socket, address):
        global tcp_recv_socket_preds
        global tcp_recv_preds_connected
        
        print("tcp receiver predictions connected.")
        if self.connected == True:
            print("WARNING: tcp prediction receiver: new incoming connection replaces existing connection!")  
        self.connected = True
        
        tcp_recv_socket_preds = socket
        tcp_recv_preds_connected = True
        
        # keep thread alive to keep socket open
        while True:
            gevent.sleep(365*24*60*60)  # wait one year
        


def connect_lsl_receiver_eeg():
    global lsl_inlet_eeg
    
    print('lsl connect receiver of EEG-Data (from NeurOne)...', end='')
    stream_infos = []
    while len(stream_infos) == 0:
        stream_infos = lsl.resolve_stream('name', LSL_RECEIVER_EEG_NAME)
    if len(stream_infos) > 1:
        print('WARNING: more than one stream from NeurOne found.')
    lsl_inlet_eeg = lsl.StreamInlet(stream_infos[0])
    print('[done]')
            

def connect_lsl_receiver_labels():
    global lsl_inlet_labels
    
    print('lsl connect receiver of labels (from UnityGame)...', end='')
    stream_infos = []
    while len(stream_infos) == 0:
        stream_infos = lsl.resolve_stream('name', LSL_RECEIVER_LABELS_NAME)
    if len(stream_infos) > 1:
        print('WARNING: more than one stream from Unity game found.')
    lsl_inlet_labels = lsl.StreamInlet(stream_infos[0])
    print('[done]')
    
            
def connect_lsl_sender_predictions():
    global lsl_outlet_predictions
    
    print('lsl start StreamOutlet for predictions...', end='')
    stream_info = lsl.StreamInfo(LSL_SENDER_PREDS_NAME, \
                                 LSL_SENDER_PREDS_TYPE, \
                                 LSL_SENDER_PREDS_STREAMCHANNELS, \
                                 lsl.IRREGULAR_RATE, \
                                 'float32', \
                                 LSL_SENDER_PREDS_ID)
    lsl_outlet_predictions = lsl.StreamOutlet(stream_info)
    print('[done]')     
    
            
def start_tcp_receiver_predictions():
    global tcp_recv_server_preds
    
    print("tcp start receiver of predictions...", end='')
    tcp_recv_server_preds = PredictionReceiveServer((TCP_RECEIVER_PREDS_HOSTNAME, \
                                                     TCP_RECEIVER_PREDS_PORT))
    tcp_recv_server_preds.start()
    print("[done]")

    
    
def connect_tcp_sender_eeg():
    global tcp_send_socket_eeg
    
    print('tcp connect sender of EEG-data (to braindecode-online)...', end='')
    connected_successfully = False
    while not connected_successfully:
        try:
            tcp_send_socket_eeg = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            tcp_send_socket_eeg.connect((TCP_SENDER_EEG_HOSTNAME, TCP_SENDER_EEG_PORT))
            connected_successfully = True
        except Exception as e:
            tcp_send_socket_eeg.close()
            # print('.', end='')
            gevent.sleep(2)
    print('[done]')
    
    gevent.sleep(2) # allow other server some time to react to connection first
    print('tcp sender eeg data: send header to braindecode-online...', end='')
    chan_line = " ".join(EEG_CHANNELNAMES) + "\n"
    tcp_send_socket_eeg.send(chan_line.encode('utf-8'))
    tcp_send_socket_eeg.send(np.array([TCP_SENDER_EEG_NCHANS], dtype=np.int32).tobytes())
    tcp_send_socket_eeg.send(np.array([TCP_SENDER_EEG_NSAMPLES/DOWNSAMPLING_COEF], dtype=np.int32).tobytes())
    print('[done]')
    

class AsyncSocketReader:    
    def __init__(self, socket):
        self.line = ''
        self.lines = []
        self.socket = socket
    
    def readlines_string(self, timeout=None, num_lines=1):
        """ Read from socket until newline reached or timeout (in seconds) is over.
            If parameter timeout=None the timeout for readlines_string is socket.gettimeout().
            If timeout is over, TimeoutError is thrown. No bytes are lost, reading can be resumed. """
        starttime = time.time()
        original_timeout = self.socket.gettimeout()
        if timeout == None:
            timeout = original_timeout
        
        while True:
            if len(self.line) == 0:
                pass
            elif self.line[-1] == '\n':
                self.lines.append(self.line)
                self.line = ''
                if len(self.lines) == num_lines:
                    self.socket.settimeout(original_timeout)
                    returnlines = self.lines
                    self.lines = []
                    return returnlines
            
            elapsed_time = time.time() - starttime
            remaining_time = timeout - elapsed_time
            if remaining_time < 0.001:
                self.socket.settimeout(original_timeout)
                raise TimeoutError
            self.socket.settimeout(remaining_time)
            try:
                self.line += self.socket.recv(1).decode('utf-8')
            except socket_timeout:
                pass



    
    
def forward_forever():
    forwarding_loopcounter = 0
    loop_time = time.time()
    fallen_behind_labels = True
    fallen_behind_eeg = True
    fallen_behind_predictions = True
    tcp_recv_socketreader_preds = AsyncSocketReader(tcp_recv_socket_preds)
    eeg_samplebuffer = np.zeros((TCP_SENDER_EEG_NCHANS, TCP_SENDER_EEG_NSAMPLES), dtype='float32')
    eeg_sample_counter = 0
    eeg_sample_label = 0
    eeg_forwarding_counter = 0
    print('Start forwarding in both directions.')
    # the operations in this loop must not block too long, to ensure that everything stays in sync
    # and we don't fall back behind the data streams
    monster_active = 0
    predictions = np.zeros((PRED_WINDOW_SIZE, 1))
    last_two_labels = [0, 0]
    while True:
        forwarding_loopcounter += 1
        wait(0.1)

        # read predictions and forward them
        #
        if forwarding_loopcounter % 2 == 0:   # save time, cause readlines_string blocks at least 1ms
            if DEBUG:
                print('read predictions from braindecode-online...')
            try:
                i_sample, preds, pred_label = tcp_recv_socketreader_preds.readlines_string(timeout=0.001, num_lines=3)
                pred_label = float(pred_label)
                last_two_labels[eeg_forwarding_counter %2] = pred_label
                if np.diff(last_two_labels) > 0:
                    monster_active = 1
                    pred_counter = 0
            except TimeoutError:
                fallen_behind_predictions = False
                if DEBUG:
                    print("no prediction to forward yet.")
            else:
                if DEBUG:
                    print('got new prediction.')
                    print('i_sample[:-1] =', i_sample[:-1])  # :-1 => without newline
                    print('preds[:-1] =', preds[:-1])
                    print('forwarding predictions.')
                splitted_predictions = preds.split(" ")
                parsed_predictions = [float(i_sample)] + \
                                    [float(splitted_predictions[i]) for i in range(PREDICTION_NUM_CLASSES)]
                if monster_active:
                    list_preds = np.array([float(splitted_predictions[0]), float(splitted_predictions[1])])
                    if np.max(list_preds) > PRED_THRESHOLD:
                        if np.argmax(list_preds) == 0:
                            predictions[pred_counter % PRED_WINDOW_SIZE] = 1
                        elif np.argmax(list_preds) == 1:
                            predictions[pred_counter % PRED_WINDOW_SIZE] = -1
                    else:
                        predictions[pred_counter % PRED_WINDOW_SIZE] = 0
                    prob = np.mean(predictions)
                    if prob < 0:
                        max_class = 2
                    else:
                        max_class = 1
                    prob = np.abs(prob)
                    prob = np.max((0, (0.8 - prob) / 0.8))
                    max_class_prob = np.array([max_class, prob], dtype='float32')
                    #lsl_outlet_predictions.push_sample(max_class_prob)
                    print('max class', max_class, 'with prob', prob)
                    pred_counter += 1

                else:
                    predictions = np.zeros((PRED_WINDOW_SIZE, 1))

                #lsl_outlet_predictions.push_sample(np.array(parsed_predictions, dtype='float32'))
                if DEBUG:
                    print('forwarding predictions done.')
                eeg_forwarding_counter += 1
    
    
def send_random_data(socket, no_blocks=100):
    print("Send random data...")
    i_block = 0  # if setting i_block to sth higher, printed results will incorrect
    max_stop_block = 10000
    stop_block = no_blocks
    cur_marker = 0
    n_samples_waiting = 49
    n_to_next_marker = n_samples_waiting
    rng = RandomState(3948394)
    assert stop_block < max_stop_block
    while i_block < stop_block:
        # chan x time
        arr = rng.randn(TCP_SENDER_EEG_NCHANS, TCP_SENDER_EEG_NSAMPLES).astype(np.float32)

        arr[-1,:] = cur_marker
        socket.send(arr.tobytes(order='F'))
        assert arr.shape == (TCP_SENDER_EEG_NCHANS, TCP_SENDER_EEG_NSAMPLES)
        i_block += 1
        gevent.sleep(0.03)
        n_to_next_marker -= TCP_SENDER_EEG_NSAMPLES
        if n_to_next_marker <= 0:
            n_to_next_marker = n_samples_waiting
            if cur_marker  == 0:
                cur_marker = float(rng.randint(1,6))
            else:
                cur_marker = 0
    
    
if __name__ == "__main__":
    gevent.signal(signal.SIGINT, gevent.kill)   # used to be signal.SIGQUIT, but didn't work
    
    #connect_lsl_receiver_eeg()
    #connect_lsl_receiver_labels()
    connect_lsl_sender_predictions()
    start_tcp_receiver_predictions()
    #connect_tcp_sender_eeg()
    
    while tcp_recv_preds_connected == False:
        gevent.sleep(0.01)
    print("Started system successfully.")
    
    forward_forever()
    

    
    
  