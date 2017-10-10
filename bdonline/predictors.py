import numpy as np

from braindecode.torch_ext.util import np_to_var, var_to_np


class DummyPredictor(object):
    def enough_data_for_new_prediction(self, buffer):
        return True

    def make_new_prediction(self, buffer):
        return buffer.n_total_samples, [buffer.marker_buffer[-1]]


class ModelPredictor(object):
    def __init__(self, model, input_time_length, pred_gap, cuda,
                 exponentiate_preds=True):
        self.model = model
        self.pred_gap = pred_gap
        self.input_time_length = input_time_length
        self.i_last_pred = -1
        self.cuda = cuda
        self.exponentiate_preds = exponentiate_preds

    def enough_data_for_new_prediction(self, buffer):
        return (buffer.n_total_samples >= self.input_time_length and
                buffer.n_total_samples > (self.i_last_pred + self.pred_gap))

    def make_new_prediction(self, buffer):
        # Compute how many samples we already have past the
        # sample we wanted to predict
        # keep in mind: n_samples = n_samples (number of samples)
        # so how many samples are we past
        # last prediction + prediction frequency
        # -1 at the end below since python  indexing is zerobased
        n_samples_after_pred = min(buffer.n_total_samples -
                                   self.input_time_length,
                                   buffer.n_total_samples - self.i_last_pred -
                                   self.pred_gap - 1)
        assert n_samples_after_pred < self.pred_gap, ("Other case "
                                                      "(multiple predictions should have happened in one "
                                                      "block that was sent) not implemented yet")
        start = -self.input_time_length - n_samples_after_pred
        end = -n_samples_after_pred
        if end == 0:
            end = None
        input_for_pred = buffer.data_buffer[start:end]
        prediction = self.predict_with_model(input_for_pred)
        # -1 since we have 0-based indexing in python
        self.i_last_pred = buffer.n_total_samples - n_samples_after_pred - 1
        return self.i_last_pred, prediction

    def predict_with_model(self, data):
        self.model.eval()
        # data is time x channels
        in_var = np_to_var(data.T[None,:,:,None], dtype=np.float32)
        if self.cuda:
            in_var = in_var.cuda()
        pred = var_to_np(self.model(in_var))
        # possibly mean across time axis
        if pred.ndim > 2:
            pred = np.mean(pred, axis=2).squeeze()
        if self.exponentiate_preds:
            pred = np.exp(pred)
        return pred


