import logging
import os.path
import time
from collections import OrderedDict

import numpy as np
import torch.nn.functional as F
from torch import optim

from braindecode.datasets.bbci import BBCIDataset
from braindecode.models.deep4 import Deep4Net
from braindecode.models.util import to_dense_prediction_model
from hyperoptim.parse import cartesian_dict_of_lists_product, \
    product_of_list_of_lists_of_dicts
from hyperoptim.util import save_pkl_artifact, save_torch_artifact

from braindecode.datasets.bcic_iv_2a import BCICompetition4Set2A
from braindecode.experiments.experiment import Experiment
from braindecode.experiments.monitors import LossMonitor, MisclassMonitor, \
    RuntimeMonitor
from braindecode.experiments.stopcriteria import MaxEpochs, NoDecrease, Or
from braindecode.datautil.iterators import BalancedBatchSizeIterator
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.datautil.splitters import split_into_two_sets
from braindecode.torch_ext.constraints import MaxNormDefaultConstraint
from braindecode.torch_ext.util import set_random_seeds, np_to_var
from braindecode.mne_ext.signalproc import mne_apply, resample_cnt
from braindecode.datautil.signalproc import (bandpass_cnt,
                                             exponential_running_standardize)
from braindecode.datautil.trial_segment import create_signal_target_from_raw_mne



log = logging.getLogger(__name__)


def get_templates():
    return {}


def get_grid_param_list():
    dictlistprod = cartesian_dict_of_lists_product
    default_params = [{
        'save_folder': './data/models/pytorch/online/niri-repl/',
        'only_return_exp': False,
    }]


    stop_params = [{
        'max_epochs': 200,
    }]

    grid_params = product_of_list_of_lists_of_dicts([
        default_params,
        stop_params,
    ])

    return grid_params


def sample_config_params(rng, params):
    return params


def run_exp(max_epochs, only_return_exp):
    from collections import OrderedDict
    filenames = ['data/robot-hall/NiRiNBD6.ds_1-1_500Hz.BBCI.mat',
                 'data/robot-hall/NiRiNBD8.ds_1-1_500Hz.BBCI.mat',
                 'data/robot-hall/NiRiNBD9.ds_1-1_500Hz.BBCI.mat',
                 'data/robot-hall/NiRiNBD10.ds_1-1_500Hz.BBCI.mat',
                 'data/robot-hall/NiRiNBD12_cursor_250Hz.BBCI.mat',
                 'data/robot-hall/NiRiNBD13_cursorS000R01_onlyFullRuns_250Hz.BBCI.mat',
                 'data/robot-hall/NiRiNBD14_cursor_250Hz.BBCI.mat',
                 'data/robot-hall/NiRiNBD15_cursor_250Hz.BBCI.mat']
    sensor_names = ['Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AF4', 'AF8', 'F7', 'F5',
                    'F3', 'F1', 'Fz', 'F2',
                    'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2',
                    'FC4', 'FC6', 'FT8',
                    'M1', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8',
                    'M2', 'TP7', 'CP5',
                    'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5',
                    'P3', 'P1', 'Pz', 'P2',
                    'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POz', 'PO4', 'PO6',
                    'PO8', 'O1', 'Oz', 'O2']
    name_to_start_codes = OrderedDict(
        [('Right Hand', [1]), ('Feet', [2]), ('Rotation', [3]), ('Words', [4])])
    name_to_stop_codes = OrderedDict(
        [('Right Hand', [10]), ('Feet', [20]), ('Rotation', [30]),
         ('Words', [40])])

    trial_ival = [500, 0]
    min_break_length_ms = 6000
    max_break_length_ms = 8000
    break_ival = [1000, -500]

    input_time_length = 700

    filename_to_extra_args = {
        'data/robot-hall/NiRiNBD12_cursor_250Hz.BBCI.mat': dict(
            name_to_start_codes=OrderedDict([('Right Hand', [1]), ('Feet', [2]),
                                             ('Rotation', [3]), ('Words', [4]),
                                             ('Rest', [5])]),
            name_to_stop_codes=OrderedDict(
                [('Right Hand', [10]), ('Feet', [20]),
                 ('Rotation', [30]), ('Words', [40]),
                 ('Rest', [50])]),
            min_break_length_ms=3700,
            max_break_length_ms=3900,
        ),
        'data/robot-hall/NiRiNBD13_cursorS000R01_onlyFullRuns_250Hz.BBCI.mat': dict(
            name_to_start_codes=OrderedDict([('Right Hand', [1]), ('Feet', [2]),
                                             ('Rotation', [3]), ('Words', [4]),
                                             ('Rest', [5])]),
            name_to_stop_codes=OrderedDict(
                [('Right Hand', [10]), ('Feet', [20]),
                 ('Rotation', [30]), ('Words', [40]),
                 ('Rest', [50])]),
            min_break_length_ms=3700,
            max_break_length_ms=3900,
        ),
        'data/robot-hall/NiRiNBD14_cursor_250Hz.BBCI.mat': dict(
            name_to_start_codes=OrderedDict([('Right Hand', [1]), ('Feet', [2]),
                                             ('Rotation', [3]), ('Words', [4]),
                                             ('Rest', [5])]),
            name_to_stop_codes=OrderedDict(
                [('Right Hand', [10]), ('Feet', [20]),
                 ('Rotation', [30]), ('Words', [40]),
                 ('Rest', [50])]),
            min_break_length_ms=3700,
            max_break_length_ms=3900,
        ),
        'data/robot-hall/NiRiNBD15_cursor_250Hz.BBCI.mat': dict(
            name_to_start_codes=OrderedDict([('Right Hand', [1]), ('Feet', [2]),
                                             ('Rotation', [3]), ('Words', [4]),
                                             ('Rest', [5])]),
            name_to_stop_codes=OrderedDict(
                [('Right Hand', [10]), ('Feet', [20]),
                 ('Rotation', [30]), ('Words', [40]),
                 ('Rest', [50])]),
            min_break_length_ms=3700,
            max_break_length_ms=3900,
        ),
    }
    from braindecode.datautil.trial_segment import \
        create_signal_target_with_breaks_from_mne
    from copy import deepcopy
    def load_data(filenames, sensor_names,
                  name_to_start_codes, name_to_stop_codes,
                  trial_ival, break_ival,
                  min_break_length_ms, max_break_length_ms,
                  input_time_length,
                  filename_to_extra_args):
        all_sets = []
        original_args = locals()
        for filename in filenames:
            kwargs = deepcopy(original_args)
            if filename in filename_to_extra_args:
                kwargs.update(filename_to_extra_args[filename])
            log.info("Loading {:s}...".format(filename))
            cnt = BBCIDataset(filename,
                              load_sensor_names=sensor_names).load()
            cnt = cnt.drop_channels(['STI 014'])
            log.info("Resampling...")
            cnt = resample_cnt(cnt, 100)
            log.info("Standardizing...")
            cnt = mne_apply(lambda a: exponential_running_standardize(a.T,
                                                                      init_block_size=50).T,
                            cnt)

            log.info("Transform to set...")
            full_set = (
                create_signal_target_with_breaks_from_mne(
                    cnt, kwargs['name_to_start_codes'],
                    kwargs['trial_ival'],
                    kwargs['name_to_stop_codes'],
                    kwargs['min_break_length_ms'],
                    kwargs['max_break_length_ms'],
                    kwargs['break_ival'],
                    prepad_trials_to_n_samples=kwargs['input_time_length'],
                ))
            all_sets.append(full_set)
        return all_sets

    sensor_names = ['Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AF4', 'AF8', 'F7', 'F5',
                    'F3', 'F1', 'Fz', 'F2',
                    'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2',
                    'FC4', 'FC6', 'FT8',
                    'M1', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8',
                    'M2', 'TP7', 'CP5',
                    'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5',
                    'P3', 'P1', 'Pz', 'P2',
                    'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POz', 'PO4', 'PO6',
                    'PO8', 'O1', 'Oz', 'O2']
    #sensor_names = ['C3', 'C4']
    n_chans = len(sensor_names)
    if not only_return_exp:
        all_sets = load_data(filenames, sensor_names,
                             name_to_start_codes, name_to_stop_codes,
                             trial_ival, break_ival,
                             min_break_length_ms, max_break_length_ms,
                             input_time_length,
                             filename_to_extra_args)
        from braindecode.datautil.signal_target import SignalAndTarget
        from braindecode.datautil.splitters import concatenate_sets

        train_set = concatenate_sets(all_sets[:6])
        valid_set = all_sets[6]
        test_set = all_sets[7]
    else:
        train_set = None
        valid_set = None
        test_set = None
    set_random_seeds(seed=20171017, cuda=True)
    n_classes = 5
    # final_conv_length determines the size of the receptive field of the ConvNet
    model = ShallowFBCSPNet(in_chans=n_chans, n_classes=n_classes,
                            input_time_length=input_time_length,
                            final_conv_length=30).create_network()
    to_dense_prediction_model(model)

    model.cuda()

    from torch import optim
    import numpy as np

    optimizer = optim.Adam(model.parameters())

    from braindecode.torch_ext.util import np_to_var
    # determine output size
    test_input = np_to_var(
        np.ones((2, n_chans, input_time_length, 1),
                dtype=np.float32))
    test_input = test_input.cuda()
    out = model(test_input)
    n_preds_per_input = out.cpu().data.numpy().shape[2]
    print("{:d} predictions per input/trial".format(n_preds_per_input))

    from braindecode.datautil.iterators import CropsFromTrialsIterator
    iterator = CropsFromTrialsIterator(batch_size=32,
                                       input_time_length=input_time_length,
                                       n_preds_per_input=n_preds_per_input)

    from braindecode.experiments.experiment import Experiment
    from braindecode.experiments.monitors import RuntimeMonitor, LossMonitor, \
        CroppedTrialMisclassMonitor, MisclassMonitor
    from braindecode.experiments.stopcriteria import MaxEpochs
    from braindecode.torch_ext.losses import log_categorical_crossentropy
    import torch.nn.functional as F
    import torch as th
    from braindecode.torch_ext.modules import Expression

    loss_function = log_categorical_crossentropy

    model_constraint = MaxNormDefaultConstraint()
    monitors = [LossMonitor(), MisclassMonitor(col_suffix='sample_misclass'),
                CroppedTrialMisclassMonitor(input_time_length),
                RuntimeMonitor(), ]
    stop_criterion = MaxEpochs(max_epochs)
    exp = Experiment(model, train_set, valid_set, test_set, iterator,
                     loss_function, optimizer, model_constraint,
                     monitors, stop_criterion,
                     remember_best_column='valid_sample_misclass',
                     run_after_early_stop=True, batch_modifier=None, cuda=True)
    if not only_return_exp:
        exp.run()

    return exp


def run(ex, max_epochs, only_return_exp, ):
    start_time = time.time()
    ex.info['finished'] = False

    exp = run_exp(max_epochs, only_return_exp)
    if not only_return_exp:
        last_row = exp.epochs_df.iloc[-1]
        end_time = time.time()
        run_time = end_time - start_time
        ex.info['finished'] = True

        for key, val in last_row.iteritems():
            ex.info[key] = float(val)
        ex.info['runtime'] = run_time
        save_torch_artifact(ex, exp.model.state_dict(), 'model_params.pkl')
        save_torch_artifact(ex, exp.model, 'model.pkl')
        save_torch_artifact(ex, exp.optimizer.state_dict(), 'trainer_params.pkl')
        save_pkl_artifact(ex, exp.epochs_df, 'epochs_df.pkl')
        save_pkl_artifact(ex, exp.before_stop_df, 'before_stop_df.pkl')
    #return exp

