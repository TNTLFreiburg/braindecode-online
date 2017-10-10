import numpy as np


def exponential_running_var_from_demeaned(demeaned_data, factor_new,
                                          start_var=None,
                                          init_block_size=None, axis=None):
    """ Compute the running var across axis 0 + given axis from demeaned data.
    For each datapoint in axis 0 its "running exponential var" is computed as:
    Its (datapoint)**2 * factor_new + so far computed var * (1-factor_new).
    You can either specify a start var or an initial block size to 
    compute the start var of. 
    In any case one var per datapoint in axis 0 is returned.
    If axis is None, no mean is computed but trial is simply used as is."""
    # TODELAY: split out if and else case into different functions
    # i.e. split apart a common function having a start value (basically the loop)
    # and then split if and else into different functions
    factor_old = 1 - factor_new
    # first preallocate the shape for the running vars for performance (otherwise much slower)
    # shape depends on which axes will be removed
    running_vars_shape = list(demeaned_data.shape)
    if axis is not None:
        for ax in axis:
            running_vars_shape.pop(ax)
    running_vars = (np.ones(running_vars_shape) * np.nan).astype(np.float32)

    if start_var is None:
        if axis is not None:
            axes_for_start_var = (0,) + axis  # also average across init trials
        else:
            axes_for_start_var = 0

        # possibly temporarily upcast to float32 to avoid overflows in sum
        # that is computed to compute mean
        start_running_var = np.mean(
            np.square(demeaned_data[0:init_block_size].astype(np.float32)),
            axis=axes_for_start_var, keepdims=True).astype(demeaned_data.dtype)
        running_vars[0:init_block_size] = start_running_var
        current_var = start_running_var
        start_i = init_block_size
    else:
        current_var = start_var
        start_i = 0

    for i in range(start_i, len(demeaned_data)):
        squared = np.square(demeaned_data[i:i + 1])
        if axis is not None:
            this_var = np.mean(squared, axis=axis, keepdims=True)
        else:
            this_var = squared
        next_var = factor_new * this_var + factor_old * current_var
        running_vars[i] = next_var
        current_var = next_var
    assert not np.any(np.isnan(running_vars))
    return running_vars


def exponential_running_mean(data, factor_new, init_block_size=None,
                             start_mean=None, axis=None):
    """ Compute the running mean across axis 0.
    For each datapoint in axis 0 its "running exponential mean" is computed as:
    Its mean * factor_new + so far computed mean * (1-factor_new).
    You can either specify a start mean or an init_block_size to 
    compute the start mean of. 
    In any case one mean per datapoint in axis 0 is returned.
    If axis is None, no mean is computed per datapoint but datapoint
    is simply used as is."""
    assert not (start_mean is None and init_block_size is None), (
        "Need either an init block or a start mean")
    assert start_mean is None or init_block_size is None, (
    "Can only use start mean "
    "or init block size")
    assert factor_new <= 1.0
    assert factor_new >= 0.0
    if isinstance(axis, int):
        axis = (axis,)
    factor_old = 1 - factor_new

    # first preallocate the shape for the running means
    # shape depends on which axes will be removed
    running_mean_shape = list(data.shape)
    if axis is not None:
        for ax in axis:
            # keep dim as empty dim
            running_mean_shape[ax] = 1

    running_means = (np.ones(running_mean_shape) * np.nan).astype(np.float32)

    if start_mean is None:
        start_data = data[0:init_block_size]
        if axis is not None:
            axes_for_start_mean = (0,) + axis  # also average across init trials
        else:
            axes_for_start_mean = 0
        # possibly temporarily upcast to float32 to avoid overflows in sum
        # that is computed to compute mean
        current_mean = np.mean(start_data.astype(np.float32),
                               keepdims=True,
                               axis=axes_for_start_mean).astype(
            start_data.dtype)
        # repeat mean for running means
        running_means[:init_block_size] = current_mean
        i_start = init_block_size
    else:
        current_mean = start_mean
        i_start = 0

    for i in range(i_start, len(data)):
        if axis is not None:
            datapoint_mean = np.mean(data[i:i + 1], axis=axis, keepdims=True)
        else:
            datapoint_mean = data[i:i + 1]
        next_mean = factor_new * datapoint_mean + factor_old * current_mean
        running_means[i] = next_mean
        current_mean = next_mean

    assert not np.any(np.isnan(running_means)), (
        "RUnning mean has NaNs :\n{:s}".format(str(running_means)))
    assert not np.any(np.isinf(running_means)), (
        "RUnning mean has Infs :\n{:s}".format(str(running_means)))
    return running_means


class StandardizeProcessor(object):
    def __init__(self, factor_new=1e-3, eps=1e-4):
        self.factor_new = factor_new
        self.eps = eps
        self.running_mean = None
        self.running_var = None

    def process(self, samples):
        if self.running_mean is not None:
            assert self.running_var is not None
            next_means = exponential_running_mean(
                samples, factor_new=self.factor_new,
                start_mean=self.running_mean)

            demeaned = samples - next_means
            next_vars = exponential_running_var_from_demeaned(
                demeaned, factor_new=self.factor_new, start_var=self.running_var)
            standardized = demeaned / np.maximum(self.eps, np.sqrt(next_vars))
            self.running_mean = next_means[-1]
            self.running_var = next_vars[-1]
            return standardized
        else:
            self.running_mean = np.mean(samples, axis=0)
            self.running_var = np.var(samples, axis=0)
            return (samples - self.running_mean) / np.maximum(
                self.eps, np.sqrt(self.running_var))


class NoProcessor(object):
    def process(self, samples):
        return samples
