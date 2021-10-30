from pathlib import Path
from preprocessing.dataprocessor import read_data

if __name__=='__main__':

    train_path = Path('datafiles/datafake_train.csv')
    test_path = Path('datafiles/datafake_test.csv')
    train_data, test_data = read_data(train_path, test_path)


    print('Data loaded')