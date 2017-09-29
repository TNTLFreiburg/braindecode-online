class ArraySupplier(object):
    def __init__(self, data, markers, block_size):
        self.data = data
        self.markers = markers
        assert len(self.data) == len(markers)
        # last new sample
        self.i_next_sample = 0
        self.block_size = block_size

    def wait_for_data(self):
        if self.i_next_sample < len(self.data):
            i_stop = min(self.i_next_sample + self.block_size, len(self.data))
            i_start = i_stop - self.block_size
            block = self.data[i_start:i_stop]
            markers = self.markers[i_start:i_stop]
            self.i_next_sample = i_stop
            return block, markers
        return None