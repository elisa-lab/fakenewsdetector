from models.supervised_models import Classifier
from preprocessing.dataprocessor import read_data, build_ngram_features
from sklearn.linear_model import LogisticRegression
import joblib
from pathlib import Path
from sklearn.metrics import classification_report


class LogisticRegressionClassifier(Classifier):
    """

    """
    def __init__(self):
        self.train_files = []
        self.test_files = []
        self.train_data = None
        self.test_data = None

        self.features = None
        self.labels = None

        self.vectorizer = None

    def add_train_file(self, train_file):
        """

        :param train_file:
        :return:
        """
        self.train_files.append(train_file)

    def add_test_file(self, test_file):
        """

        :param test_file:
        :return:
        """
        self.test_files.append(test_file)

    def load_data(self):
        """

        :return:
        """
        train_data, test_data = read_data(self.train_files, self.test_files)
        self.train_data = train_data
        self.test_data = test_data

    def prepare_data(self):
        """

        :return:
        """
        self.vectorizer = build_ngram_features(self.train_data)

        # convert training corpus in to vectors
        self.features = self.vectorizer.transform(self.train_data['post']).toarray()
        # extracting the output labels
        self.labels = self.train_data['fake']

    def train_model(self):
        """

        :return:
        """
        self.model = LogisticRegression()
        self.model.fit(self.features, self.labels)

    def save_model(self, model_path):
        """

        :param model_path:
        :return:
        """
        joblib.dump(self.model, Path(model_path / 'lr.joblib.pkl'), compress=9)

    def load_model(self, model_path):
        """

        :param model_path:
        :return:
        """
        self.model = joblib.load(Path(model_path / 'lr.joblib.pkl'))

    def predict_model(self):
        """

        :return:
        """
        features = self.vectorizer.transform(self.test_data['post'])
        predictions = self.model.predict(features)
        return predictions

    def calculate_accuracy(self):
        """

        :return:
        """
        y_pred = self.predict_model()
        y_true = self.test_data['fake']

        print(classification_report(y_true=y_true, y_pred=y_pred))
