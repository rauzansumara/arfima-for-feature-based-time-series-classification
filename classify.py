# import packages
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
import utils
import pandas as pd
from os import path

from sktime.series_as_features.compose import FeatureUnion
from sktime.transformations.panel.reduce import Tabularizer
from sktime.datatypes._panel._convert import from_nested_to_2d_array
from sktime.transformations.panel.tsfresh import TSFreshFeatureExtractor

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler


def tabularisation_features(Xtrain: pd.DataFrame, Ytrain: pd.DataFrame, Xtest: pd.DataFrame, Ytest: pd.DataFrame):

    # tabularisation feature extraction
    X_train_tab = from_nested_to_2d_array(Xtrain)
    X_test_tab = from_nested_to_2d_array(Xtest)

    # define classifiers
    classifiers_list = [
        [make_pipeline(MinMaxScaler(), KNeighborsClassifier(n_neighbors=1)), 'k-NN (k=1)'],
        [make_pipeline(MinMaxScaler(), KNeighborsClassifier(n_neighbors=5)), 'k-NN (k=5)'],
        [make_pipeline(MinMaxScaler(), SVC(gamma='scale', kernel='rbf')), 'SVM'],
        [make_pipeline(MinMaxScaler(), RandomForestClassifier(n_estimators=500)), 'RF'],
        [make_pipeline(MinMaxScaler(), AdaBoostClassifier(n_estimators=500)), 'AdaBoost'],
    ]

    # looping for prediction
    for classifier in classifiers_list:
        classifier[0].fit(X_train_tab, Ytrain)
        y_test_pred = classifier[0].predict(X_test_tab)
        ACC = accuracy_score(Ytest, y_test_pred)
        print(f"{classifier[1]}, ACC = {ACC:.4f}")
    

def arfima_features(Xtrain: pd.DataFrame, Ytrain: pd.DataFrame, Xtest: pd.DataFrame, Ytest: pd.DataFrame):
    
    # combine feature extraction
    combined_features = FeatureUnion([
                                    ('arfima', utils.RowTransformer(FunctionTransformer(func=utils.arfima_coefs, validate=False))),
                                    ('others', utils.RowTransformer(FunctionTransformer(func=utils.other_features, validate=False)))
                                ])
    
    # fit feature extraction
    comb_X_train = combined_features.fit_transform(Xtrain)
    comb_X_test = combined_features.fit_transform(Xtest)

    # define classifiers
    classifiers_list = [
        [make_pipeline(Tabularizer(), MinMaxScaler(), KNeighborsClassifier(n_neighbors=1)), 'k-NN (k=1)'],
        [make_pipeline(Tabularizer(), MinMaxScaler(), KNeighborsClassifier(n_neighbors=5)), 'k-NN (k=5)'],
        [make_pipeline(Tabularizer(), MinMaxScaler(), SVC(gamma='scale', kernel='rbf')), 'SVM'],
        [make_pipeline(Tabularizer(), MinMaxScaler(), RandomForestClassifier(n_estimators=500)), 'RF'],
        [make_pipeline(Tabularizer(), MinMaxScaler(), AdaBoostClassifier(n_estimators=500)), 'AdaBoost'],
    ]

    # looping for prediction
    for classifier in classifiers_list:
        classifier[0].fit(comb_X_train, Ytrain)
        y_test_pred = classifier[0].predict(comb_X_test)
        ACC = accuracy_score(Ytest, y_test_pred)
        print(f"{classifier[1]}, ACC = {ACC:.4f}")


def run(data_name: str):

    X_train, y_train = utils.load_from_arff_to_dataframe(path.join("datasets", data_name, f"{data_name}_TRAIN.arff"))
    X_test, y_test = utils.load_from_arff_to_dataframe(path.join("datasets", data_name, f"{data_name}_TEST.arff"))

    print(f'Tabularisation -> classifying {data_name} dataset:')
    tabularisation_features(X_train, y_train, X_test, y_test)

    print(f'ARFIMA -> classifying {data_name} dataset:')
    arfima_features(X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SyntaxError("Usage: python3 classify.py data_name")
    run(sys.argv[1])
