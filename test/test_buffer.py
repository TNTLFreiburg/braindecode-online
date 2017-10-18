import numpy as np

from bdonline.buffer import DataMarkerBuffer


def test_buffer():
    buffer = DataMarkerBuffer(2, 3)
    buffer.buffer([[1, -1], [2, -2]], [1, 2])
    assert np.testing.assert_array_equal(buffer.data_buffer, [[0, 0], [1, -1], [2, -2]])
    assert np.testing.assert_array_equal(buffer.marker_buffer, [0, 1, 2])
    buffer.buffer([[3, -3], [4, -4]], [3, 4])
    assert np.testing.assert_array_equal(buffer.data_buffer, [[2, -2], [3, -3], [4, -4]])
    assert np.testing.assert_array_equal(buffer.marker_buffer, [2, 3, 4])