import datetime
from glob import glob
import os.path
import logging
from copy import deepcopy
import time

import numpy as np
from numpy.random import RandomState

from braindecode.datautil.signalproc import exponential_running_standardize
from braindecode.datautil.iterators import _get_start_stop_blocks_for_trial, \
    _create_batch_from_i_trial_start_stop_blocks
from braindecode.torch_ext.util import var_to_np, np_to_var

log = logging.getLogger(__name__)


class NoTrainer(object):
    def new_data_available(self, buffer, marker_block):
        return

    def set_n_chans(self, n_chans):
        return

    def add_data_from_today(self, data_processor):
        return


class BatchCntTrainer(object):
    def __init__(self, model,
                 loss_function, model_loss_function, model_constraint,
                 optimizer, input_time_length,
                 n_preds_per_input,
                 n_classes,
                 n_updates_per_break, batch_size,
                 n_min_trials, trial_start_offset, break_start_offset,
                 break_stop_offset, add_breaks=True,
                 min_break_samples=0, min_trial_samples=0,
                 cuda=True):
        self.__dict__.update(locals())
        del self.self
        self.rng = RandomState(30948348)
        self.data_batches = []
        self.y_batches = []


    def set_n_chans(self, n_chans):
        test_input = np_to_var(
            np.ones((2, n_chans, self.input_time_length, 1),
                    dtype=np.float32))
        if self.cuda:
            test_input = test_input.cuda()
        out = self.model(test_input)
        self.n_preds_per_input = int(out.size()[2])
        self.n_classes = int(out.size()[1])
        return


    def add_data_from_today(self, factor_new, eps):
        # Check if old data exists, if yes add it
        now = datetime.datetime.now()
        day_string = now.strftime('%Y-%m-%d')
        data_folder = 'data/online/{:s}'.format(day_string)
        # sort should sort timewise for our timeformat...
        data_files = sorted(glob(os.path.join(data_folder, '*.npy')))
        if len(data_files) > 0:
            log.info("Loading {:d} data files for adaptation:\n{:s}".format(
                len(data_files), str(data_files)))
            for filename in data_files:
                log.info("Add data from {:s}...".format(filename))
                samples_markers = np.load(filename)
                samples = samples_markers[:, :-1]
                markers = np.int32(samples_markers[:, -1])
                self.add_training_blocks_from_old_data(
                    samples, markers, factor_new=factor_new,
                    eps=eps)
            log.info(
                "Done loading, now have {:d} trials (including breaks)".format(
                    len(self.data_batches)))
        else:
            log.info(
                "No data files found to load for adaptation in {:s}".format(
                    data_folder))

    def add_training_blocks_from_old_data(self, old_samples,
                                          old_markers, factor_new, eps):
        # first standardize data
        old_samples = exponential_running_standardize(old_samples,
                                                      factor_new=factor_new,
                                                      init_block_size=1000,
                                                      eps=eps)
        trial_starts, trial_stops = self.get_trial_start_stop_indices(
            old_markers)
        log.info("Adding {:d} trials".format(len(trial_starts)))
        for trial_start, trial_stop in zip(trial_starts, trial_stops):
            self.add_trial(trial_start, trial_stop,
                           old_samples, old_markers)
        # now lets add breaks
        log.info("Adding {:d} breaks".format(len(trial_starts) - 1))
        for break_start, break_stop in zip(trial_stops[:-1],
                                           trial_starts[1:]):
            self.add_break(break_start, break_stop, old_samples,
                           old_markers)

    def new_data_available(self, buffer, marker_block):
        # Check if a trial has ended with last samples
        # need marker samples with one overlap
        # so we do not miss trial boundaries inbetween two sample blocks
        marker_buffer = buffer.marker_buffer
        marker_samples_with_overlap = np.copy(
            marker_buffer[-len(marker_block) - 1:])
        trial_has_ended = np.sum(
            np.diff(marker_samples_with_overlap) < 0) > 0
        if trial_has_ended:
            trial_starts, trial_stops = self.get_trial_start_stop_indices(
                marker_buffer)
            trial_start = trial_starts[-1]
            trial_stop = trial_stops[-1]
            ## Logging and assertions
            log.info("Trial has ended for class {}, from {} to {}".format(
                int(marker_buffer[trial_start]), trial_start, trial_stop))
            assert trial_start < trial_stop, ("trial start {} should be "
                                              "before trial stop {}, markers: {}").format(
                trial_start,
                trial_stop, str(marker_samples_with_overlap))
            assert marker_buffer[trial_start - 1] == 0, (
                "Expect a 0 marker before trial start, instead {}".format(
                    marker_buffer[trial_start - 1]))
            assert marker_buffer[trial_start] != 0, (
                "Expect a nonzero marker at trial start instead {}".format(
                    marker_buffer[trial_start]))
            assert marker_buffer[trial_stop - 1] != 0, (
                "Expect a nonzero marker at trial end instead {}".format(
                    marker_buffer[trial_stop]))
            assert marker_buffer[trial_start] == marker_buffer[
                trial_stop - 1], (
                "Expect a same marker at trial start and end instead {} / {}".format(
                    marker_buffer[trial_start],
                    marker_buffer[trial_stop]))
            self.add_trial(trial_start, trial_stop,
                           buffer.data_buffer,
                           marker_buffer)
            log.info("Now {} trials (including breaks)".format(
                len(self.data_batches)))

            start_train_time = time.time()
            self.train()
            end_train_time = time.time()
            log.info("Time for training: {:.2f}s".format(
                end_train_time - start_train_time))

        trial_has_started = np.sum(
            np.diff(marker_samples_with_overlap) > 0) > 0
        if trial_has_started:
            trial_end_in_marker_buffer = np.sum(
                np.diff(marker_buffer) < 0) > 0
            if trial_end_in_marker_buffer:
                # +1 necessary since diff removes one index
                trial_start = \
                np.flatnonzero(np.diff(marker_buffer) > 0)[-1] + 1
                trial_stop = \
                np.flatnonzero(np.diff(marker_buffer) < 0)[-1] + 1
                assert trial_start > trial_stop, (
                "If trial has just started "
                "expect this to be after stop of last trial")
                assert marker_buffer[trial_start - 1] == 0, (
                    "Expect a 0 marker before trial start, instead {:d}".format(
                        marker_buffer[trial_start - 1]))
                assert marker_buffer[trial_start] != 0, (
                    "Expect a nonzero marker at trial start instead {:d}".format(
                        marker_buffer[trial_start]))
                self.add_break(break_start=trial_stop,
                               break_stop=trial_start,
                               all_samples=buffer.data_buffer,
                               all_markers=marker_buffer)
                # log.info("Break added, now at {:d} batches".format(len(self.data_batches)))

    def add_trial(self, trial_start, trial_stop, all_samples, all_markers):
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
        if (pred_start < trial_stop) and (
                trial_stop - trial_start >= self.min_trial_samples):
            assert (
            len(np.unique(all_markers[pred_start:trial_stop])) == 1), (
                "All predicted markers should be the same in one trial, instead got:"
                "{:s}".format(
                    str(np.unique(all_markers[trial_start:trial_stop]))))
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
                           all_markers):
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
        assert len(all_markers) == len(all_samples)
        assert pred_stop < len(all_markers)
        needed_samples = all_samples[in_sample_start:pred_stop]
        needed_markers = np.copy(all_markers[pred_start:pred_stop])
        needed_markers = np.concatenate((np.zeros(crop_size - 1,
                                                  dtype=needed_markers.dtype),
                                         needed_markers))
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

    def add_trial_topo_trial_y(self, trial_data, trial_markers):
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
            trial_data, trial_y, start_stop_blocks, self.n_preds_per_input)
        self.data_batches.append(batch[0])
        self.y_batches.append(batch[1])

    def _create_batch_from_start_stop_blocks(self, trial_X, trial_y,
                                             start_stop_blocks,
                                             n_preds_per_input):
        Xs = []
        ys = []
        for start, stop in start_stop_blocks:
            Xs.append(trial_X[start:stop].T[:,:,None])
            ys.append(trial_y[stop - n_preds_per_input:stop])
        batch_X = np.array(Xs).astype(np.float32)
        batch_y = np.array(ys).astype(np.int64)
        return batch_X, batch_y

    def train(self):
        n_trials = len(self.data_batches)
        if n_trials >= self.n_min_trials:
            log.info("Training model...")
            # Remember values as backup in case of NaNs
            model_param_dict_before = deepcopy(self.model.state_dict())
            optimizer_dict_before = deepcopy(self.optimizer.state_dict())

            all_y_blocks = np.concatenate(self.y_batches, axis=0)

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
                i_supercrops = self.rng.choice(n_total_supercrops,
                                               size=self.batch_size,
                                               p=block_probs)
                this_y = np.asarray(all_y_blocks[i_supercrops])
                this_topo = np.zeros((len(i_supercrops),) +
                                     self.data_batches[0].shape[1:],
                                     dtype=np.float32)
                for i_batch_row, i_supercrop in enumerate(i_supercrops):
                    i_global_batch, i_global_row = \
                    i_supercrop_to_batch_and_row[i_supercrop]
                    supercrop_data = self.data_batches[i_global_batch][
                        i_global_row]
                    this_topo[i_batch_row] = supercrop_data
                self.train_on_batch(this_topo, this_y)

            any_nans = np.any([np.any(np.isnan(var_or_tensor_to_np(v)))
                               for v in self.model.state_dict().values()])
            if any_nans:
                log.warning("Reset train parameters due to NaNs")
                self.optimizer.load_state_dict(optimizer_dict_before)
                self.model.load_state_dict(model_param_dict_before)
            any_nans = np.any([np.any(np.isnan(var_or_tensor_to_np(v)))
                               for v in self.model.state_dict().values()])
            assert not any_nans

        else:
            log.info(
                "Not training model yet, only have {:d} of {:d} trials ".format(
                    n_trials, self.n_min_trials))

    def train_on_batch(self, inputs, targets):
        self.model.train()
        input_vars = np_to_var(inputs)
        target_vars = np_to_var(targets)
        if self.cuda:
            input_vars = input_vars.cuda()
            target_vars = target_vars.cuda()
        self.optimizer.zero_grad()
        outputs = self.model(input_vars)
        loss = self.loss_function(outputs, target_vars)
        if self.model_loss_function is not None:
            loss = loss + self.model_loss_function(self.model)
        loss.backward()
        self.optimizer.step()
        if self.model_constraint is not None:
            self.model_constraint.apply(self.model)



def var_or_tensor_to_np(v):
    try:
        return var_to_np(v)
    except RuntimeError:
        return v.cpu().numpy()