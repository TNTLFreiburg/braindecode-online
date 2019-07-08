
''' This module has as a goal to send indexes to the braindecode server to mimick the arrival of real data

'''
import sys
import numpy as np
import gevent
from psychopy.clock import wait
import gevent.server
from gevent.server import StreamServer
import gevent.select

from gevent import socket

TCP_SENDER_EEG_PORT = 7987                          # port of braindecode-online
TCP_SENDER_EEG_HOSTNAME = '127.0.0.1'               # hostname of braindecode-online
TCP_SENDER_EEG_NCHANS = 65                          # number of channels to send to braindecode-online, includes labels
TCP_SENDER_EEG_NSAMPLES = 10                       # number of samples per channel to send to braindecode-online at once
BYTEORDER = sys.byteorder
EEG_CHANNELNAMES = ['Fp1', 'Fpz', 'Fp2', 'AF7',     # channel names send to braindecode-online
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
TCP_RECEIVER_PREDS_HOSTNAME = 'localhost'           # hostname of this PC, used to receive predictions
TCP_RECEIVER_PREDS_PORT = 30000


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
            gevent.sleep(365 * 24 * 60 * 60)  # wait one year



def start_tcp_receiver_predictions():
    global tcp_recv_server_preds

    print("tcp start receiver of predictions...", end='')
    tcp_recv_server_preds = PredictionReceiveServer((TCP_RECEIVER_PREDS_HOSTNAME, \
                                                     TCP_RECEIVER_PREDS_PORT))
    tcp_recv_server_preds.start()
    print("[done]")
    return tcp_recv_server_preds
def main():
    tcp_recv_server_preds =  start_tcp_receiver_predictions()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((TCP_RECEIVER_PREDS_HOSTNAME, TCP_SENDER_EEG_PORT))
        idx = np.array([0])
        while True:
            sock.sendall(idx.tobytes())
            idx +=1
            wait(0.004)

if __name__=='__main__':
    main()