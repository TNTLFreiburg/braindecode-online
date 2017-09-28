from bdonline.storer import DataMarkerPredictionStorer

class OnlineExperiment(object):
    def __init__(self, supplier, buffer, processor, predictor,
                 trainer):
        self.supplier = supplier
        self.buffer = buffer
        self.processor = processor
        self.storer = DataMarkerPredictionStorer()
        self.predictor = predictor
        self.trainer = trainer

    def run(self):
        while self.supplier.will_have_more_data():
            self.run_one_iteration()

    def run_one_iteration(self):
        data, markers = self.supplier.wait_for_data()
        data = self.processor.process(data)
        self.buffer.buffer(data, markers)
        self.storer.store_data_markers(data, markers)
        if self.predictor.enough_data_for_new_prediction(self.buffer):
            i_sample, prediction = self.predictor.make_new_prediction(
                self.buffer)
            self.storer.store_prediction(i_sample, prediction)
        self.trainer.new_data_available(self.buffer)

