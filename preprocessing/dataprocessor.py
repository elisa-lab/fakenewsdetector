"""
This script will read the input data and preprocess it. Also, it will covert into features.
"""
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from typing import List


def read_data(train_files: List, test_files: List):
    """

    :param train_files:
    :param test_files:
    :return:
    """
    train_data = []
    for train_file in train_files:
        train_data.append(pd.read_csv(train_file, delimiter=';'))

    train_data = pd.concat(train_data, ignore_index=True)

    test_data = []
    for test_file in test_files:
        test_data.append(pd.read_csv(test_file, delimiter=';'))

    test_data = pd.concat(test_data, ignore_index=True)

    return train_data, test_data


def build_ngram_features(train_data):
    """

    :param train_data:
    :return:
    """
    final_stopwords_list = stopwords.words('french')
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 3), stop_words=final_stopwords_list,
                                 max_features=5000)
    vectorizer.fit(train_data['post'])
    return vectorizer
