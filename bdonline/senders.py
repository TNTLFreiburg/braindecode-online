import logging

log = logging.getLogger(__name__)

class PrintSender(object):
    def send_prediction(self, i_sample, prediction):
        print("Sending prediction {:d}:\n{:s}".format(i_sample,
                                                      str(prediction)))


class NoSender(object):
    def send_prediction(self, i_sample, prediction):
        pass


class PredictionSender(object):
    def __init__(self, socket):
        self.socket = socket

    def send_prediction(self, i_sample, prediction, label):
        log.info("Prediction for sample {:d}:\n{:s} with true label \n {}".format(
            i_sample, str(prediction), str(label)))
        # +1 to convert 0-based to 1-based indexing
        self.socket.sendall("{:d}\n".format(i_sample + 1).encode('utf-8'))
        n_preds = len(prediction) # e.g. number of classes
        # format all preds as floats with spaces inbetween
        format_str = " ".join(["{:f}"] * n_preds) + "\n"
        pred_str = format_str.format(*prediction)
        self.socket.sendall(pred_str.encode('utf-8'))