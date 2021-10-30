"""
This script will read the input data and preprocess it. Also, it will covert into features.
"""
import pandas as pd

def read_data(train_file, test_file):
    train_data = pd.read_csv(train_file, delimiter=';')
    test_data = pd.read_csv(test_file, delimiter=';')

    return train_data, test_data