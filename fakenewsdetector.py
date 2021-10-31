from pathlib import Path
from preprocessing.dataprocessor import read_data, build_ngram_features

if __name__=='__main__':

    train_path = Path('datafiles/datafake_train.csv')
    test_path = Path('datafiles/datafake_test.csv')
    train_data, test_data = read_data(train_path, test_path)
    vectorizer = build_ngram_features(train_data)




    print('Data loaded')