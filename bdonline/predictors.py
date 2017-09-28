
class DummyPredictor(object):
    def enough_data_for_new_prediction(self, buffer):
        return True

    def make_new_prediction(self, buffer):
        return buffer.n_total_samples, buffer.marker_buffer[-1]
