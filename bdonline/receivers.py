import numpy as np
from bdonline.read import read_until_bytes_received_or_enter_pressed



class DataReceiver(object):
    def __init__(self, socket, n_chans, n_samples_per_block, n_bytes_per_block, savetimestamps):
        self.socket = socket
        self.n_chans = n_chans
        self.n_samples_per_block = n_samples_per_block
        self.n_bytes_per_block = n_bytes_per_block
        self.savetimestamps = savetimestamps

    def wait_for_data(self):
        array = read_until_bytes_received_or_enter_pressed(
            self.socket, self.n_bytes_per_block)
        if array is None:
            return None
        else:
            if self.savetimestamps:
                array = np.fromstring(array, dtype=np.float32)
                array = array.reshape(self.n_chans, self.n_samples_per_block,
                                  order='F')
                data = array[:-2].T
                markers = array[-2]
                timestamps = array[-1]
                return data, markers, timestamps
            else:
                array = np.fromstring(array, dtype=np.float32)
                array = array.reshape(self.n_chans, self.n_samples_per_block, order='F')
                data = array[:-1].T
                markers = array[-1]
                return data, markers
