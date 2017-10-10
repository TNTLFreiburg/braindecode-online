import numpy as np

class RingBuffer(np.ndarray):
    'A multidimensional ring buffer.'

    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return

    def __array_wrap__(self, out_arr, context=None):
        return np.ndarray.__array_wrap__(self, out_arr, context)

    def extend(self, xs):
        'Adds array xs to the ring buffer. If xs is longer than the ring '
        'buffer, the last len(ring buffer) of xs are added the ring buffer.'
        xs = np.asarray(xs)
        if self.shape[1:] != xs.shape[1:]:
            raise ValueError("Element's shape mismatch. RingBuffer.shape={}. "
                             "xs.shape={}".format(self.shape, xs.shape))
        len_self = len(self)
        len_xs = len(xs)
        if len_self <= len_xs:
            xs = xs[-len_self:]
            len_xs = len(xs)
        else:
            self[:-len_xs] = self[len_xs:]
        self[-len_xs:] = xs

    def append(self, x):
        'Adds element x to the ring buffer.'
        x = np.asarray(x)
        if self.shape[1:] != x.shape:
            raise ValueError("Element's shape mismatch. RingBuffer.shape={}. "
                             "xs.shape={}".format(self.shape, x.shape))
        self[:-1] = self[1:]
        self[-1] = x


class DataMarkerBuffer(object):
    def __init__(self, n_chans, n_samples_in_buffer):
        self.data_buffer = RingBuffer(np.zeros((
            n_samples_in_buffer, n_chans), dtype=np.float32))
        self.marker_buffer = RingBuffer(np.zeros((
            n_samples_in_buffer), dtype=np.float32))
        self.n_total_samples = 0

    def buffer(self, data, markers):
        assert len(data) == len(markers)
        self.data_buffer.extend(data)
        self.marker_buffer.extend(markers)
        self.n_total_samples += len(data)
