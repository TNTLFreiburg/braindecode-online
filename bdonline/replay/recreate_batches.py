import datetime
from glob import glob
import os.path
import glob
import re
import logging
from copy import deepcopy
import time
import torch
import numpy as np
import xdf_to_bd
from numpy.random import RandomState
from torch.autograd import Variable
from braindecode.models.util import to_dense_prediction_model
from braindecode.models import deep4
from braindecode.datautil.signalproc import exponential_running_standardize
from braindecode.datautil.iterators import _get_start_stop_blocks_for_trial, \
    _create_batch_from_i_trial_start_stop_blocks
from braindecode.torch_ext.util import var_to_np, np_to_var

log = logging.getLogger(__name__)
trial_start_offset = 500
fs = 250
ms_to_samples = fs / 1000.0

base_name = 'D:\\braindecode-online\\bdonline\\models\\best_model'
model_name = os.path.join(base_name, 'deep_4_600')
model_dict = torch.load(model_name)
final_conv_length = 2
input_time_length = 500
n_chans =64
n_classes=2
model = deep4.Deep4Net(n_chans, n_classes, input_time_length, final_conv_length)
model = model.create_network()
model.load_state_dict(model_dict)
to_dense_prediction_model(model)
model.cuda()

batch_folder = "D:\\DLVR\\savegrad\\"
all_batch_files = glob.glob(batch_folder + "batch*")


def my_key(file):
    string_numbers = re.search(r'\d_\d(.*)', file).group()
    split_string_numbers = string_numbers.split("-")
    hour = split_string_numbers[0][2:]
    minutes = split_string_numbers[1]
    seconds =  split_string_numbers[2]
    batch_number =  split_string_numbers[3]
    return eval(hour) * 3600 + eval(minutes) * 60 + eval(seconds) +eval(batch_number)

all_batch_files = sorted(all_batch_files, key=my_key)
all_supercrops = (torch.load(file) for file in all_batch_files)
for supercrop in all_supercrops:
    print(max(supercrop))
all_supercrops = [torch.load(file) for file in all_batch_files]

def set_n_chans(model, n_chans=64, input_time_length=600):
    test_input = np_to_var(
        np.ones((2, n_chans, input_time_length, 1),
                dtype=np.float32))
    test_input = test_input.cuda()
    out = model(test_input)
    n_preds_per_input = int(out.size()[2])
    n_classes = int(out.size()[1])
    return n_preds_per_input, n_classes


class BatchCreator(object):
    def __init__(self, path, files, input_time_length,  n_preds_per_input, n_classes,
                 n_updates_per_break=5, batch_size=45,
                 n_min_trials=4, trial_start_offset=0 * ms_to_samples, break_start_offset=1000 * ms_to_samples,
                 break_stop_offset=-1000 * ms_to_samples,
                 min_break_samples=0, min_trial_samples=0,
                 add_breaks=False, savetimestamps=False):
        self.__dict__.update(locals())
        del self.self
        self.rng = RandomState(30948348)
        self.data_labels = xdf_to_bd.dlvr_braindecode(path, files, -1, 250).T
        if savetimestamps:
            self.data_buffer = self.data_labels[:, :-2]
            self.marker_buffer = self.data_labels[:, -2]
            self.timestamp_buffer = self.data_labels[:, -1]
        else:
            self.data_buffer = self.data_labels[:, :-1]
            self.marker_buffer = self.data_labels[:, -1]
        self.data_batches = []
        self.y_batches = []
        self.savetimestamps = savetimestamps
        self.timestamp_batches = []
        self.n_preds_per_input = n_preds_per_input
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.trial_start_offset = int(trial_start_offset)
        self.break_start_offset = int(break_start_offset)
        self.break_stop_offset = int(break_stop_offset)
        self.add_breaks = add_breaks
        self.min_break_samples = min_break_samples
        self.min_trial_samples = min_trial_samples
        self.n_min_trials = n_min_trials
        self.supercrop_counter = 0

    def new_data_available(self):
        # Check for trials in the loaded data_buffer
        # need marker samples with one overlap
        # so we do not miss trial boundaries inbetween two sample blocks
        trial_has_ended = np.sum(
            np.diff(self.marker_buffer) < 0) > 0
        if trial_has_ended:
            trial_starts, trial_stops = self.get_trial_start_stop_indices(
                self.marker_buffer)
            for trial_start, trial_stop in zip(trial_starts, trial_stops):
                ## Logging and assertions
                log.info("Trial has ended for class {}, from {} to {}".format(
                    int(self.marker_buffer[trial_start]), trial_start, trial_stop))
                assert trial_start < trial_stop, ("trial start {} should be "
                                                  "before trial stop {}, markers: {}").format(
                    trial_start,
                    trial_stop, str(self.marker_buffer))
                assert self.marker_buffer[trial_start - 1] == 0, (
                    "Expect a 0 marker before trial start, instead {}".format(
                        self.marker_buffer[trial_start - 1]))
                assert self.marker_buffer[trial_start] != 0, (
                    "Expect a nonzero marker at trial start instead {}".format(
                        self.marker_buffer[trial_start]))
                assert self.marker_buffer[trial_stop - 1] != 0, (
                    "Expect a nonzero marker at trial end instead {}".format(
                        self.marker_buffer[trial_stop]))
                assert self.marker_buffer[trial_start] == self.marker_buffer[
                    trial_stop - 1], (
                    "Expect a same marker at trial start and end instead {} / {}".format(
                        self.marker_buffer[trial_start],
                        self.marker_buffer[trial_stop]))
                if self.savetimestamps:
                    self.add_trial(trial_start, trial_stop,
                                   self.data_buffer,
                                   self.marker_buffer, self.timestamp_buffer)
                else:                    self.add_trial(trial_start, trial_stop,
                                   self.data_buffer,
                                   self.marker_buffer)
                log.info("Now {} trials (including breaks)".format(
                    len(self.data_batches)))
                start_train_time = time.time()
                self.train()
                end_train_time = time.time()
                log.info("Time for training: {:.2f}s".format(
                    end_train_time - start_train_time))
            if self.add_breaks:
                trial_has_started = np.sum(
                    np.diff(self.marker_buffer) > 0) > 0
                if trial_has_started:
                    trial_end_in_marker_buffer = np.sum(
                        np.diff(self.marker_buffer) < 0) > 0
                    if trial_end_in_marker_buffer:
                        # +1 necessary since diff removes one index
                        trial_start = \
                            np.flatnonzero(np.diff(self.marker_buffer) > 0)[-1] + 1
                        trial_stop = \
                            np.flatnonzero(np.diff(self.marker_buffer) < 0)[-1] + 1
                        assert trial_start > trial_stop, (
                            "If trial has just started "
                            "expect this to be after stop of last trial{}")
                        assert self.marker_buffer[trial_start - 1] == 0, (
                            "Expect a 0 marker before trial start, instead {:d}".format(
                                self.marker_buffer[trial_start - 1]))
                        assert self.marker_buffer[trial_start] != 0, (
                            "Expect a nonzero marker at trial start instead {:d}".format(
                                self.marker_buffer[trial_start]))
                        self.add_break(break_start=trial_stop,
                                       break_stop=trial_start,
                                       all_samples=self.data_buffer,
                                       all_markers=self.marker_buffer)
                        # log.info("Break added, now at {:d} batches".format(len(self.data_batches)))

    def add_trial(self, trial_start, trial_stop, all_samples, all_markers, all_time_stamps=np.array([])):
        # Add trial by determining needed signal/samples and markers
        # In case the model predicts more samples concurrently
        # than the number of trials in this sample
        # prepad the markers with -1 and signal with zeros
        assert (len(np.unique(all_markers[trial_start:trial_stop])) == 1), (
            "All markers should be the same in one trial, instead got:"
            "{:s}".format(
                str(np.unique(all_markers[trial_start:trial_stop]))))
        # determine markers and in samples for default case
        pred_start = trial_start + self.trial_start_offset
        print(pred_start)
        if (pred_start < trial_stop) and (
                        trial_stop - trial_start >= self.min_trial_samples):
            assert (
                len(np.unique(all_markers[pred_start:trial_stop])) == 1), (
                "All predicted markers should be the same in one trial, instead got:"
                "{:s}".format(
                    str(np.unique(all_markers[trial_start:trial_stop]))))
            if all_time_stamps.any():
                self.add_trial_or_break(pred_start, trial_stop, all_samples,
                                    all_markers, all_time_stamps)
            else:
                self.add_trial_or_break(pred_start, trial_stop, all_samples,
                                    all_markers)
        elif pred_start >= trial_stop:
            log.warning(
                "Prediction start {:d} is past trial stop {:d}".format(
                    pred_start, trial_stop) + ", not adding trial")
        else:
            assert trial_stop - trial_start < self.min_trial_samples
            log.warn(
                "Trial only {:d} samples, want {:d} samples, not using.".format(
                    trial_stop - trial_start, self.min_trial_samples))

    def add_break(self, break_start, break_stop, all_samples, all_markers):
        if self.add_breaks:
            all_markers = np.copy(all_markers)
            assert np.all(all_markers[break_start:break_stop] == 0), (
                "all markers in break should be 0, instead have markers:"
                "{:s}\nbreak start: {:d}\nbreak stop: {:d}\nmarker sequence: {:s}".format(
                    str(np.unique(all_markers[break_start:break_stop]
                                  )), break_start, break_stop,
                    str(all_markers[break_start - 1:break_stop + 1])))
            assert all_markers[break_start - 1] != 0
            assert all_markers[break_stop] != 0
            pred_start = break_start + self.break_start_offset
            pred_stop = break_stop + self.break_stop_offset
            if (pred_start < pred_stop) and (
                            break_stop - break_start >= self.min_break_samples):
                # keep n_classes for 1-based matlab indexing logic in markers
                all_markers[pred_start:pred_stop] = self.n_classes
                self.add_trial_or_break(pred_start, pred_stop, all_samples,
                                        all_markers)
            elif pred_start >= pred_stop:
                log.warning(
                    "Prediction start {:d} is past prediction stop {:d}".format(
                        pred_start, pred_stop) + ", not adding break")
            else:
                assert break_stop - break_start < self.min_break_samples
                log.warn(
                    "Break only {:d} samples, want {:d} samples, not using.".format(
                        break_stop - break_start, self.min_break_samples))

        else:
            pass  # Ignore break that was supposed to be added

    def add_trial_or_break(self, pred_start, pred_stop, all_samples,
                           all_markers, all_timestamps=np.array([])):
        """Assumes all markers already changed the class for break."""
        crop_size = self.input_time_length - self.n_preds_per_input + 1
        if pred_stop < crop_size:
            log.info("Prediction stop {:d} smaller crop size {:d}, not adding"
                     "".format(pred_stop, crop_size))
            return
        in_sample_start = pred_start - crop_size + 1
        if in_sample_start < 0:
            pred_start = crop_size - 1
            in_sample_start = pred_start - crop_size + 1
            assert in_sample_start == 0
            log.info("Prediction start before start of buffer, resetting to 0.")
        assert in_sample_start >= 0
        # Later functions need one marker per input sample
        # (so also need markers at start that will not actually be used,
        # which are only there
        # for the receptive field of ConvNet)
        # These functions will then cut out correct markers.
        # We want to make sure that no unwanted markers are used, so
        # we only extract the markers that will actually be predicted
        # and pad with 0s (which will be converted to -1 (!) and
        # should not be used later, except
        # trial too small and we go into the if clause below)
        print('len_markers, len_sample', len(all_markers), len(all_samples))
        assert len(all_markers) == len(all_samples)
        assert pred_stop < len(all_markers)
        needed_samples = all_samples[in_sample_start:pred_stop]
        needed_markers = np.copy(all_markers[pred_start:pred_stop])
        needed_markers = np.concatenate((np.zeros(crop_size - 1,
                                                  dtype=needed_markers.dtype),
                                         needed_markers))
        if all_timestamps.any():
            needed_timestamps = all_timestamps[in_sample_start:pred_stop]
        assert len(needed_samples) == len(needed_markers), (
            "{:d} markers and {:d} samples (should be same)".format(
                len(needed_samples), len(needed_markers)))
        n_expected_samples = pred_stop - pred_start + crop_size - 1
        # this assertion here for regression reasons, failed before
        assert len(needed_markers) == n_expected_samples, (
            "Extracted {:d} markers, but should have {:d}".format(
                len(needed_markers), n_expected_samples))
        # handle case where trial is too small
        if pred_stop - pred_start < self.n_preds_per_input:
            log.warn("Trial/break has only {:d} predicted samples "
                     "from {:d} to {:d} in it, "
                     "less than the "
                     "{:d} concurrently processed samples of the model! "
                     "Will add padding that should be masked during training.".format(
                pred_stop - pred_start,
                pred_start, pred_stop,
                self.n_preds_per_input))
            # add -1 markers that will not be used during training for the
            # data before
            n_pad_samples = self.n_preds_per_input - (pred_stop - pred_start)
            pad_markers = np.zeros(n_pad_samples, dtype=all_markers.dtype) - 1
            needed_markers = np.concatenate((pad_markers, needed_markers))
            pad_samples = np.zeros_like(all_samples[0:n_pad_samples])
            needed_samples = np.concatenate((pad_samples, needed_samples))
            pred_start = pred_start - n_pad_samples

        assert pred_stop - pred_start >= self.n_preds_per_input
        n_expected_samples = pred_stop - pred_start + crop_size - 1
        assert len(needed_markers) == n_expected_samples, (
            "Extracted {:d} markers, but should have {:d}".format(
                len(needed_markers), n_expected_samples))
        assert len(needed_samples) == n_expected_samples, (
            "Extracted {:d} samples, but should have {:d}".format(
                len(needed_samples), n_expected_samples))
        if all_timestamps.any():
            self.add_trial_topo_trial_y(needed_samples, needed_markers, needed_timestamps)
        else:
            self.add_trial_topo_trial_y(needed_samples, needed_markers)


    def get_trial_start_stop_indices(self, markers):
        # + 1 as diff "removes" one index, i.e. diff will be above zero
        # at the index 1 before the increase=> the trial start
        trial_starts = np.flatnonzero(np.diff(markers) > 0) + 1
        # diff removing index, so this index is last sample of trial
        # but stop indices in python are exclusive so +1
        trial_stops = np.flatnonzero(np.diff(markers) < 0) + 1

        if len(trial_starts) == 0 or len(trial_stops) == 0:
            print('len(trial_starts):', len(trial_starts))
            print('len(trial_stops):', len(trial_stops))

        if trial_starts[0] >= trial_stops[0]:
            # cut out first trial which only has end marker
            trial_stops = trial_stops[1:]
        if trial_starts[-1] >= trial_stops[-1]:
            # cut out last trial which only has start marker
            trial_starts = trial_starts[:-1]

        assert (len(trial_starts) == len(trial_stops)), (
            "Have {:d} trial starts, but {:d} trial stops (should be equal)".format(
                len(trial_starts), len(trial_stops)))
        assert (np.all(trial_starts <= trial_stops))
        return trial_starts, trial_stops

    def add_trial_topo_trial_y(self, trial_data, trial_markers, timestamps=np.array([])):
        """ needed_samples are samples needed for predicting entire trial,
        i.e. they typically include a part before the first sample of the trial."""
        crop_size = self.input_time_length - self.n_preds_per_input + 1
        assert (len(np.unique(trial_markers[(crop_size - 1):])) == 1) or (
            (len(np.unique(trial_markers[(crop_size - 1):])) == 2) and (
                0 in trial_markers[(crop_size - 1):])), (
            ("Trial should have exactly one class, markers: {:s} ").format(
                np.unique(trial_markers[(crop_size - 1):])))
        trial_y = np.copy(
            trial_markers) - 1  # -1 as zero is non-trial marker
        trial_len = len(trial_data)
        i_pred_start = crop_size - 1
        i_pred_stop = trial_len
        start_stop_blocks = _get_start_stop_blocks_for_trial(
            i_pred_start, i_pred_stop, self.input_time_length,
            self.n_preds_per_input)
        assert start_stop_blocks[0][0] == 0, (
            "First block should start at first sample, starts at {:d}".format(
                start_stop_blocks[0][0]
            ))
        batch = self._create_batch_from_start_stop_blocks(
            trial_data, trial_y, start_stop_blocks, self.n_preds_per_input, timestamps)
        self.data_batches.append(batch[0])
        self.y_batches.append(batch[1])
        if timestamps.any():
            self.timestamp_batches.append(batch[2])

    def _create_batch_from_start_stop_blocks(self, trial_X, trial_y,
                                             start_stop_blocks,
                                             n_preds_per_input, timestamps=np.array([])):
        Xs = []
        ys = []
        if timestamps.any():
            Ts = []
        for start, stop in start_stop_blocks:
            Xs.append(trial_X[start:stop].T[:, :, None])
            ys.append(trial_y[stop - n_preds_per_input:stop])
            if timestamps.any():
                Ts.append(timestamps[start:stop])
        batch_X = np.array(Xs).astype(np.float32)
        batch_y = np.array(ys).astype(np.int64)
        if timestamps.any():
            batch_T = np.array(Ts)
            return batch_X, batch_y, batch_T
        else:
            return batch_X, batch_y

    def train(self):
        n_trials = len(self.data_batches)
        if n_trials >= self.n_min_trials:
            log.info("Training model...")
            # Remember values as backup in case of NaNs

            all_y_blocks = np.concatenate(self.y_batches, axis=0)
            if self.savetimestamps:
                all_timestamps_blocks = np.concatenate(self.timestamp_batches)
            # make classes balanced
            # hopefully this is correct?! any sample shd be fine, -1 is a random decision
            labels_per_block = all_y_blocks[:, -1]

            # Rebalance by calculating frequencies of classes in data
            # and then rebalancing by sampling with inverse probability
            unique_labels = sorted(np.unique(labels_per_block))
            class_probs = {}
            for i_class in unique_labels:
                freq = np.mean(labels_per_block == i_class)
                prob = 1.0 / (len(unique_labels) * freq)
                class_probs[i_class] = prob
            block_probs = np.zeros(len(labels_per_block))
            for i_class in unique_labels:
                block_probs[labels_per_block == i_class] = class_probs[
                    i_class]
            # Renormalize probabilities
            block_probs = block_probs / np.sum(block_probs)

            # Create mapping from super crop nr -> data batch nr, row nr

            n_rows_per_batch = [len(b) for b in self.data_batches]
            n_total_supercrops = np.sum(n_rows_per_batch)
            print("we have this many supercrops", n_total_supercrops)
            assert n_total_supercrops == len(all_y_blocks)
            i_supercrop_to_batch_and_row = np.zeros((n_total_supercrops, 2),
                                                    dtype=np.int32)
            i_batch = 0
            i_batch_row = 0
            for i_supercrop in range(n_total_supercrops):
                if i_batch_row == n_rows_per_batch[i_batch]:
                    i_batch_row = 0
                    i_batch += 1
                i_supercrop_to_batch_and_row[i_supercrop][0] = i_batch
                i_supercrop_to_batch_and_row[i_supercrop][1] = i_batch_row
                i_batch_row += 1

            assert i_batch == len(n_rows_per_batch) - 1
            assert i_batch_row == n_rows_per_batch[-1]

            for _ in range(self.n_updates_per_break):
                i_supercrops = all_supercrops[self.supercrop_counter]
                random_crops = self.rng.choice(n_total_supercrops,size=self.batch_size,p=block_probs)
                print("what we randomly got", random_crops, len(random_crops))
                print("during the experiment", all_supercrops[self.supercrop_counter])
                this_topo = np.zeros((len(i_supercrops),) +
                                     self.data_batches[0].shape[1:],
                                     dtype=np.float32)
                for i_batch_row, i_supercrop in enumerate(i_supercrops):
                    i_global_batch, i_global_row = \
                    i_supercrop_to_batch_and_row[i_supercrop]
                    supercrop_data = self.data_batches[i_global_batch][
                        i_global_row]
                    this_topo[i_batch_row] = supercrop_data
                now = datetime.datetime.now()
                file_name = str(now.day) + '-' + str(now.month) + '-' + str(now.year) + '_' + \
                            str(now.hour) + '-' + str(now.minute) + '-' + str(now.second) + str(_)
                torch.save(this_topo, 'D:\\Data-replay\\batched_data\\' + 'batches' + file_name + '.pkl')
                self.supercrop_counter += 1
        else:
            log.info(
                "Not training model yet, only have {:d} of {:d} trials ".format(
                    n_trials, self.n_min_trials))
