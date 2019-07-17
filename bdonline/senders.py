import json

from kafka import KafkaProducer

from bdonline.server import log


class PrintSender(object):
    def send_prediction(self, i_sample, prediction):
        print("Sending prediction {:d}:\n{:s}".format(i_sample,
                                                      str(prediction)))


class NoSender(object):
    def send_prediction(self, i_sample, prediction):
        pass


class KafkaSender(object):
    def __init__(self, bootstrap_servers='172.30.0.119:32768', topic_name='decoding_feedback'):
        self.bootstrap_servers = bootstrap_servers
        self.topic_name = topic_name
        self.producer = KafkaProducer(bootstrap_servers=self.bootstrap_servers, value_serializer=lambda m: json.dumps(
            m).encode('ascii'))

    def send_prediction(self, i_sample, prediction):
        print("Sending prediction {:d}:\n{:s}".format(i_sample, str(prediction)))
        self.producer.send(self.topic_name, {'sample': i_sample, 'prediction': prediction})


class PredictionSender(object):
    def __init__(self, socket):
        self.socket = socket

    def send_prediction(self, i_sample, prediction):
        log.info("Prediction for sample {:d}:\n{:s}".format(
            i_sample, str(prediction)))
        # +1 to convert 0-based to 1-based indexing
        self.socket.sendall("{:d}\n".format(i_sample + 1).encode('utf-8'))
        n_preds = len(prediction)  # e.g. number of classes
        # format all preds as floats with spaces inbetween
        format_str = " ".join(["{:f}"] * n_preds) + "\n"
        pred_str = format_str.format(*prediction)
        self.socket.sendall(pred_str.encode('utf-8'))
