class PrintSender(object):
    def send_prediction(self, i_sample, prediction):
        print("Sending prediction {:d}:\n{:s}".format(i_sample,
                                                      str(prediction)))


class NoSender(object):
    def send_prediction(self, i_sample, prediction):
        pass
