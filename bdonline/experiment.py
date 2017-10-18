from bdonline.storer import DataMarkerPredictionStorer

class OnlineExperiment(object):
    def __init__(self, supplier, buffer, processor, predictor,
                 trainer, sender):
        self.supplier = supplier
        self.buffer = buffer
        self.processor = processor
        self.storer = DataMarkerPredictionStorer()
        self.predictor = predictor
        self.trainer = trainer
        self.sender = sender

    def set_supplier(self, supplier):
        self.supplier = supplier

    def set_buffer(self, buffer):
        self.buffer = buffer

    def set_sender(self, sender):
        self.sender = sender

    def run(self):
        supplier_at_end_of_data = False
        while not supplier_at_end_of_data:
            supplier_at_end_of_data = self.run_one_iteration()

    def run_one_iteration(self):
        data_and_markers = self.supplier.wait_for_data()
        supplier_at_end_of_data = data_and_markers is None
        if not supplier_at_end_of_data:
            self.run_one_iteration_on_data(data_and_markers[0],
                                           data_and_markers[1])
        return supplier_at_end_of_data

    def run_one_iteration_on_data(self, data, markers):
        # store directly so you can even test changes of preprocessing later...
        self.storer.store_data_markers(data, markers)
        data = self.processor.process(data)
        self.buffer.buffer(data, markers)
        if self.predictor.enough_data_for_new_prediction(self.buffer):
            i_sample, prediction = self.predictor.make_new_prediction(
                self.buffer)
            self.storer.store_prediction(i_sample, prediction)
            self.sender.send_prediction(i_sample, prediction)
        self.trainer.new_data_available(self.buffer, markers)

