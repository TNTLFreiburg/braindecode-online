import numpy as np
class DataMarkerPredictionStorer(object):
    def __init__(self):
        self.data_blocks = []
        self.marker_blocks = []
        self.timestamp_blocks = []
        self.i_samples_and_predictions = []

    def store_data_markers(self, data, markers, timestamps=np.array([])):
        self.data_blocks.append(data)
        self.marker_blocks.append(markers)
        if timestamps.any():
            self.timestamp_blocks.append(timestamps)

    def store_prediction(self, i_sample, prediction):
        self.i_samples_and_predictions.append((i_sample, prediction))
