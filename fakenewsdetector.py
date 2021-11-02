from pathlib import Path

from models.logistic_regression import LogisticRegressionClassifier

if __name__=='__main__':
    # setting data paths
    train_path = Path('datafiles/datafake_train.csv')
    test_path = Path('datafiles/datafake_test.csv')
    model_path = Path('save_models/')

    # initialiaze the class
    lr_model = LogisticRegressionClassifier()

    # adding the train and test files to the model
    lr_model.add_train_file(train_path)
    lr_model.add_test_file(test_path)

    # loading the data from the files.
    lr_model.load_data()

    # extracting features from the data
    lr_model.prepare_data()

    # training the model using extracted features and labels
    lr_model.train_model()

    lr_model.save_model(model_path)
    lr_model.load_model(model_path)

    # calculate accuracy on the test data
    lr_model.calculate_accuracy()

