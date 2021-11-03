from models.supervised_models import Classifier
from preprocessing.dataprocessor import read_data, build_ngram_features
from sklearn.tree import DecisionTreeClassifier

import joblib
from pathlib import Path
from sklearn.metrics import classification_report
from typing import List


class DecisionTreeClassifier(Classifier):

    def __init__(self):
        self.train_files = []
        self.test_files = []
        self.train_data = None
        self.test_data = None

        self.features = None
        self.labels = None

        self.vectorizer = None

    def add_train_file(self, train_file: Path) -> None:
        """
        Adds a train file to the train file list

        :param train_file: Path of the train file
        """
        self.train_files.append(train_file)

    def add_test_file(self, test_file: Path) -> None:
        """
        Adds a test file to the test file list

        :param test_file: Path of the test file
        """
        self.test_files.append(test_file)

    def load_data(self) -> None:
        """
        Loads the train and test data
        """
        train_data, test_data = read_data(self.train_files, self.test_files)
        self.train_data = train_data
        self.test_data = test_data

    def prepare_data(self) -> None:
        """
        Extracts the features from the dataset (ngram and more)
        """
        self.vectorizer = build_ngram_features(self.train_data)

        # convert training corpus in to vectors
        self.features = self.vectorizer.transform(self.train_data['post']).toarray()
        # extracting the output labels
        self.labels = self.train_data['fake']

    def train_model(self) -> None:
        """
        Trains a model using the prepared dataset
        """
        self.model = DecisionTreeClassifier()
        self.model.fit(self.features, self.labels)

    def save_model(self, model_path: Path) -> None:
        """
       Saves the trained model

       :param model_path: Path of the dictionary where the model will be saved
        """

        joblib.dump(self.model, Path(model_path / 'lr.joblib.pkl'), compress=9)

    def load_model(self, model_path: Path) -> None:
        """
        Loads the trained model

        :param model_path: Path of the dictionary from where the model will be loaded
        """
        self.model = joblib.load(Path(model_path / 'lr.joblib.pkl'))

    def predict_model(self) -> List:
        """
        Predicts the output of the test data

        :return: Returns predictions on the test data by the model
        """
        features = self.vectorizer.transform(self.test_data['post']).toarray()
        predictions = self.model.predict(features)

        return predictions

    def calculate_accuracy(self) -> None:
        """
        Calculate accuracy on the test data
        """
        y_pred = self.predict_model()
        y_true = self.test_data['fake']

        print(classification_report(y_true=y_true, y_pred=y_pred))
