import pyxdf as xdf
import numpy as np
from scipy.signal import filtfilt, iirnotch, butter

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
        file = xdf.load_xdf(file_name)
        stream_names = np.array([item['info']['name'][0] for item in file[0]])
        eeg_idx = list(stream_names).index('NeuroneStream')
        game_idx = list(stream_names).index('Game State')
        eeg_time_series = file[0][eeg_idx]['time_series'].T  # Transpose to micmick incoming stream
        eeg_time_stamps = file[0][eeg_idx]['time_stamps']
        monster_side, monster_side_times, monster_destroyed_times = monster_status(file, game_idx)
        eeg_time_series = applyfilters_downsample(1, 30, target_fs, eeg_time_series)
        labels = label_maker(eeg_time_stamps, monster_side, monster_side_times, monster_destroyed_times)
        eeg_time_series = np.concatenate([eeg_time_series[:34], eeg_time_series[42:-3]])
        eeg_time_series = np.concatenate([eeg_time_series, labels])
        total_experiment.append(eeg_time_series)

    total_experiment = np.concatenate(total_experiment, axis=1)
    return total_experiment, eeg_channel_names

def main():
    xdf_filenames=['D:\\DLVR\\Data\\subjH\\block1.xdf', 'D:\\DLVR\\Data\\subjH\\block2.xdf', 'D:\\DLVR\\Data\\subjH\\block3.xdf']
    DATA_AND_LABELS, CHAN_NAMES = bd_data_from_xdf(xdf_filenames, 250)
    return DATA_AND_LABELS

if __name__=='__main__':
    main()