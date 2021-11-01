class Classifier:
    '''
    Abstract class for traditional supervised models
    '''

    def add_train_file(self, train_file):
        raise NotImplementedError()

    def add_test_file(self, test_file):
        raise NotImplementedError()

    def load_data(self):
        raise NotImplementedError()

    def prepare_data(self):
        raise NotImplementedError()

    def train_model(self):
        raise NotImplementedError()

    def save_model(self, model_path):
        raise NotImplementedError()

    def load_model(self, model_path):
        raise NotImplementedError()

    def predict_model(self):
        raise NotImplementedError()



