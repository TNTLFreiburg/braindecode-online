import numpy as np

from bdonline import ArraySupplier

def test_array_supplier():
    supplier = ArraySupplier(
        np.array((np.arange(9), np.arange(0,-9,-1))).T, np.arange(10,19), 3)
    assert supplier.i_next_sample == 0
    data, marker = supplier.wait_for_data()
    assert np.testing.assert_array_equal(data, [[0,0],[1,-1],[2,-2]])
    assert np.testing.assert_array_equal(marker, [10,11,12])
    assert supplier.i_next_sample == 3
    data, marker = supplier.wait_for_data()
    assert np.testing.assert_array_equal(data, [[3,-3],[4,-4],[5,-5]])
    assert np.testing.assert_array_equal(marker, [13,14,15])
    assert supplier.i_next_sample == 6
    data, marker = supplier.wait_for_data()
    assert np.testing.assert_array_equal(data, [[6,-6],[7,-7],[8,-8]])
    assert np.testing.assert_array_equal(marker, [16,17,18])
    assert supplier.i_next_sample == 9
    data_and_marker = supplier.wait_for_data()
    assert data_and_marker is None
    assert supplier.i_next_sample == 9