"""
This script will read the input data and preprocess it. Also, it will covert into features.
"""
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords


def read_data(train_file, test_file):
    train_data = pd.read_csv(train_file, delimiter=';')
    test_data = pd.read_csv(test_file, delimiter=';')

    return train_data, test_data

def build_ngram_features(train_data):
    final_stopwords_list = stopwords.words('french')
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(1,3), stop_words=final_stopwords_list)
    vectorizer.fit(train_data['post'])
    return vectorizer
