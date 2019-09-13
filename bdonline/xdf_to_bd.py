import sys
import os
sys.path.append('/home/matthias/xdf_mne_interface')
sys.path.append('D:\\DLVR\\braindecode')


import pyxdf as xdf
import resampy
import numpy as np
from scipy.signal import filtfilt, iirnotch, butter
import mne_interface as mni
from braindecode.datautil.signalproc import exponential_running_standardize, exponential_running_demean

def preprocess(eeg_data):
    """Returns the eeg data filtered, downsampled and exponentially standardized"""
    # filter signal
    B_1, A_1 = butter(5, 1, btype='high', output='ba', fs=5000)

    # Butter filter (lowpass) for 30 Hz
    B_40, A_40 = butter(6, 120, btype='low', output='ba', fs=5000)

    # Notch filter with 50 HZ
    F0 = 50.0
    Q = 30.0  # Quality factor

    # Design notch filter
    B_50, A_50 = iirnotch(F0, Q, 5000)

    eeg_data = filtfilt(B_50, A_50, eeg_data)
    eeg_data = filtfilt(B_40, A_40, eeg_data)
    eeg_data = resampy.resample(eeg_data, 5000, 250, axis=1)
    eeg_data = eeg_data.astype(np.float32)
    eeg_data = exponential_running_standardize(eeg_data.T, factor_new=0.001, init_block_size=None,
                                    eps=0.0001).T
    return eeg_data
    
	
def applyfilters_downsample(filter1freq, filter2freq, target_fs, eeg_time_series):
        b_1, a_1 = butter(5, filter1freq, btype='high', output='ba', fs = target_fs)
        b_30, a_30 = butter(6, filter2freq, btype='low', output='ba', fs = target_fs)
        f0 = 50.0
        Q = 30.0  # Quality factor
        # Design notch filter
        b_50, a_50 = iirnotch(f0, Q, target_fs)
        eeg_time_series = filtfilt(b_50, a_50, eeg_time_series)
        eeg_time_series = filtfilt(b_30, a_30, eeg_time_series)
        eeg_time_series = filtfilt(b_1, a_1, eeg_time_series)
        eeg_time_series = eeg_time_series[:, ::20]
        return eeg_time_series

def monster_status(file, game_idx):
    monster_side = []
    monster_side_idx = []
    monster_destroyed_idx = []
    game_time_stamps = file[0][game_idx]['time_stamps']
    for idx, state in enumerate(file[0][game_idx]['time_series']):
        if state == ['Monster right']:
            monster_side.append('Monster right')
            monster_side_idx.append(idx)
        elif state == ['Monster left']:
            monster_side.append('Monster left')
            monster_side_idx.append(idx)
        elif state == ['Monster destroyed']:
            monster_destroyed_idx.append(idx)
    monster_side_times = game_time_stamps[monster_side_idx]
    monster_destroyed_times = game_time_stamps[monster_destroyed_idx]
    return monster_side, monster_side_times, monster_destroyed_times



def label_maker(eeg_time_stamps, monster_side,  monster_side_times, monster_destroyed_times):
    labels = []
    for timestamp in eeg_time_stamps:
        if len(monster_side_times) == 0:
            labels.append(0)
        elif timestamp < monster_side_times[0]:
            labels.append(0)
        elif monster_side_times[0] < timestamp < monster_destroyed_times[0]:
            if monster_side[0] == 'Monster right':
                labels.append(1)
            elif monster_side[0] == 'Monster left':
                labels.append(2)
        elif timestamp > monster_side_times[0] and timestamp > monster_destroyed_times[0]:
            monster_side = monster_side[1:]
            monster_destroyed_times = monster_destroyed_times[1:]
            monster_side_times = monster_side_times[1:]
            labels.append(0)
    labels = np.array(labels).reshape(1, -1)
    labels = labels[:, ::20]
    return labels

def bd_data_from_xdf(xdf_file_names, target_fs):
    '''Function takes in a list containing names for xdf files to be read in
    returns the data in the format that bdonline expects, channels x time'''
    total_experiment = []
    eeg_channel_names = ['Fp1', 'Fpz', 'Fp2', 'AF7',  # channel names send to braindecode-online
                         'AF3', 'AF4', 'AF8', 'F7',
                         'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5',
                         'FC3',
                         'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'M1', 'T7', 'C5',
                         'C3',
                         'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'M2', 'TP7', 'CP5', 'CP3',
                         'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3',
                         'P1',
                         'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POz',
                         'PO4',
                         'PO6', 'PO8', 'O1', 'Oz', 'O2', 'marker']
    for file_name in xdf_file_names:
        print('Reading ', file_name)
        file = xdf.load_xdf(file_name)
        stream_names = np.array([item['info']['name'][0] for item in file[0]])
        eeg_idx = list(stream_names).index('NeuroneStream')
        game_idx = list(stream_names).index('Game State')
        eeg_time_series = file[0][eeg_idx]['time_series'].T  # Transpose to micmick incoming stream
        eeg_time_stamps = file[0][eeg_idx]['time_stamps']
        monster_side, monster_side_times, monster_destroyed_times = monster_status(file, game_idx)
        eeg_time_series = preprocess(eeg_time_series)
        labels = label_maker(eeg_time_stamps, monster_side, monster_side_times, monster_destroyed_times)
        eeg_time_series = np.concatenate([eeg_time_series[:32], eeg_time_series[40:-3]])
        print(eeg_time_series.shape, labels.shape)
        eeg_time_series = np.concatenate([eeg_time_series, labels], axis=0)
        total_experiment.append(eeg_time_series)

    total_experiment = np.concatenate(total_experiment, axis=1)
    return total_experiment, eeg_channel_names


def dlvr_braindecode(path, files, timeframe_start, target_fps, emg=False):
    """ Uses event markers to extract motor tasks from multiple DLVR .xdf files.
    Args:
        path: If the files share a single path, you can specify it here.

        files: A list of .xdf files to extract data from.

        timeframe_start: The time in seconds before the event, in which the EEG Data is extracted.

        target_fps: Downsample the EEG-data to this value.

    Returns:
        X: A list of trials
        y: An array specifying the action
    """

    # Epochs list containing differently sized arrays [#eeg_electrodes, times]
    X = []
    breaks =[]
    # event ids corresponding to the trials where 'left' = 0 and 'right' = 1
    y = np.array([])
    for file in files:
        # load a file
        print('Reading ', file)
        current_raw = mni.xdf_loader(path + file)
        if emg:
            # For MEGVR experiments switch EMG into C3/4
            current_raw._data[14, :] = current_raw.get_data(picks=['EMG_LH'])  # C3
            current_raw._data[16, :] = current_raw.get_data(picks=['EMG_RH'])  # C4

        # discard EOG/EMG
        current_raw.pick_types(meg=False, eeg=True)
        # pick only relevant events
        events = current_raw.events[(current_raw.events[:, 2] == current_raw.event_id['Monster left']) | (
        current_raw.events[:, 2] == current_raw.event_id['Monster right'])]
        # timestamps where a monster deactivates
        stops = current_raw.events[:, 0][(current_raw.events[:, 2] == current_raw.event_id['Monster destroyed'])]
        # timestamps where trials begin
        starts = events[:, 0]

        # extract event_ids and shift them to [0, 1]
        key = events[:, 2]
        key = (key == key.max()).astype(np.int64)
        key += 1
        # standardize, convert to size(time, channels)
        # current_raw._data = exponential_running_standardize(current_raw._data.T, factor_new=0.001, init_block_size=None, eps=0.0001).T

        # Find the trials and their corresponding end points
        for count, event in enumerate(starts):
            # in case the last trial has no end (experiment ended before the trial ends), discard it

            # Get the trial from 1 second before the task starts to the next 'Monster deactived' flag
            if count == 0:
                current_break = current_raw._data[:, :event - round(timeframe_start * 5000)]
            else:
                current_break = current_raw._data[:, stops[stops < event][-1] : event - round(timeframe_start * 5000)]
            current_epoch = current_raw._data[:, event - round(timeframe_start * 5000): stops[stops > event][0]]

            current_epoch = preprocess(current_epoch)
            current_break = preprocess(current_break)

            label_row = np.ones((1, current_epoch.shape[1])) * key[count]
            current_epoch = np.concatenate((current_epoch, label_row), axis=0)

            current_break = np.concatenate((current_break, np.zeros((1, current_break.shape[1]))), axis=0)

            X.append(current_epoch)
            breaks.append(current_break)

    data_labels = np.concatenate((breaks[0], X[0]), axis=1)
    for break_time, trial in zip(breaks[1:], X[1:]):
        data_labels = np.concatenate((data_labels, break_time), axis=1)
        data_labels = np.concatenate((data_labels, trial), axis=1)

    end_break = current_raw._data[:, stops[-1]:]
    end_break = preprocess(end_break)
    end_break = np.concatenate((end_break, np.zeros((1, end_break.shape[1]))), axis=0)
    data_labels = np.concatenate((data_labels, end_break), axis=1)
    return data_labels


def main():
    xdf_filenames=['D:\\DLVR\\Data\\subjH\\block1.xdf', 'D:\\DLVR\\Data\\subjH\\block2.xdf', 'D:\\DLVR\\Data\\subjH\\block3.xdf']
    DATA_AND_LABELS, CHAN_NAMES = bd_data_from_xdf(xdf_filenames, 250)
    return DATA_AND_LABELS

if __name__=='__main__':
    main()
