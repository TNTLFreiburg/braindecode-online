class ArraySupplier(object):
    """
    Supplier for given data.
    
    Parameters
    ----------
    data: 2d-array
        time x channels
    markers: 1d-array
        time
    block_size: int
    """
    def __init__(self, data, markers, block_size):
        self.data = data
        self.markers = markers
        assert len(self.data) == len(markers)
        # last new sample
        self.i_next_sample = 0
        self.block_size = block_size

    def wait_for_data(self):
        i_stop = self.i_next_sample + self.block_size
        if  i_stop <= len(self.data):
            i_start = self.i_next_sample
            block = self.data[i_start:i_stop]
            markers = self.markers[i_start:i_stop]
            self.i_next_sample = i_stop
            return block, markers
        return None