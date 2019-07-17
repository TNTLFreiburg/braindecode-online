import json

import numpy as np
from kafka import KafkaConsumer

from bdonline.read import read_until_bytes_received_or_enter_pressed


class DataReceiver(object):
    def __init__(self, socket, n_chans, n_samples_per_block, n_bytes_per_block):
        self.socket = socket
        self.n_chans = n_chans
        self.n_samples_per_block = n_samples_per_block
        self.n_bytes_per_block = n_bytes_per_block

    def wait_for_data(self):
        array = read_until_bytes_received_or_enter_pressed(
            self.socket, self.n_bytes_per_block)
        if array is None:
            return None
        else:
            array = np.fromstring(array, dtype=np.float32)
            array = array.reshape(self.n_chans, self.n_samples_per_block,
                                  order='F')
            data = array[:-1].T
            markers = array[-1]
            return data, markers


class KafkaReceiver(object):
    def __init__(self, bootstrap_servers='172.30.0.119:32768', topic_name='bd-online_input'):
        self.bootstrap_servers = bootstrap_servers
        self.topic_name = topic_name
        self.consumer = KafkaConsumer(self.topic_name, bootstrap_servers=self.bootstrap_servers,
                                      value_deserializer=lambda m: json.loads(m.decode('ascii')))

    def wait_for_data(self):
        message = self.consumer.poll()
        if message == {}:
            return None
        else:
            data = []
            markers = []
            topic_key = [*message.keys()][0]
            for consumer_record in message[topic_key]:
                value = consumer_record.value
                print('{}'.format(value))
                for key in message.keys():
                    data.append(value[key])
                    markers.append(key)

        return data, markers
