from functools import partial
from pathlib import Path
from typing import Dict
import pandas as pd
from sklearn.datasets import make_classification
# Data processing
import pandas as pd
import yaml
import numpy as np
from collections import Counter
# Visualization
import matplotlib.pyplot as plt
# Model and performance
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import PredefinedSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import sklearn
from sklearn.svm import SVC
from xgboost import XGBClassifier
import pickle
from utilsfile import read_csv, to_csv
import Classifier.FeatureReader as FeatureReader
from Classifier.FeatureReader import get_reader
from Classifier.ClfLogger import logger
from consts.global_consts import  ROOT_PATH, DATA_PATH_INTERACTIONS, NEGATIVE_DATA_PATH, MERGE_DATA, DATA_PATH_INTERACTIONS
from sklearn.ensemble import IsolationForest

class NoModelFound(Exception):
    pass

class ClassifierWithGridSearch(object):
    def __init__(self, dataset_file, result_dir,number_iteration):
        self.dataset_file = dataset_file
        self.dataset_name = self.extract_dataset_name()
        print(f"Handling dataset : {self.dataset_name}")
        self.load_dataset()
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(exist_ok=True, parents=True)
        self.create_clf_dict()

    def extract_dataset_name(self):
        return str(self.dataset_file.stem).split("_test")[0]

    def create_clf_dict(self):
        self.clf_dict = {
            "isolation_forest": IsolationForest(),
        }

    # this function response on load the dataset
    def load_dataset(self):

        directory = self.dataset_file.parent
        feature_reader = get_reader()
        X, y = feature_reader.file_reader(directory / f"{self.dataset_name}.csv")
        self.X = X
        self.y = y



     # this function response on train model and then save this model
    def train_one_conf(self, clf_name, conf,number_iteration, scoring="accuracy"):

        output_file = self.result_dir / f"{self.dataset_name}_method_{number_iteration}_{clf_name}.csv"

        # creat the specific clf and load the parameters of the clf according to the ymal file.
        clf = self.clf_dict[clf_name]
        print(clf)
        parameters = conf['parameters']

        # this step find to optimize prams
        grid_obj = GridSearchCV(clf, parameters, scoring=scoring, cv=4, n_jobs=-1, verbose=3)
        grid_obj.fit(self.X, self.y)

        print('\n Best estimator:')
        print(grid_obj.best_estimator_)
        print(grid_obj.best_score_ * 2 - 1)
        # save the best classifier
        best_clf = grid_obj.best_estimator_
        model_file = self.result_dir / f"{self.dataset_name}_method_{number_iteration}_{clf_name}.model"

        try:
            with model_file.open("wb") as pfile:
                pickle.dump(best_clf, pfile)
        except Exception:
            pass
        results = pd.DataFrame(grid_obj.cv_results_)
        results.to_csv(output_file, index=False)

        # def fit_best_clf(self, clf_name):
        #     clf = self.clf_dict[clf_name]
        #     print(clf)
        #     fit_params = {}
        #     if clf_name == "xgbs":
        #         fit_params = {"eval_set": [(self.X, self.y)],
        #                       "early_stopping_rounds": 50}
        #
        #     clf.fit(self.X, self.y, **fit_params)
        #     clf.save_model("model_sklearn.json")
        #
        #     return clf

    def fit(self, yaml_path,number_iteration):
        with open(yaml_path, 'r') as stream:
            training_config = yaml.safe_load(stream)

        for clf_name, conf in training_config.items():
            key_classifier = (list(self.clf_dict.keys())[0])
            if conf["run"] and clf_name == key_classifier:
                self.train_one_conf(clf_name, conf,number_iteration, scoring="accuracy")


def worker(dataset_file, results_dir, yaml_file, number_iteration):
    clf_grid_search = ClassifierWithGridSearch(dataset_file=dataset_file, result_dir=results_dir,number_iteration=number_iteration)
    clf_grid_search.fit(yaml_file,number_iteration)
    return


def self_fit(feature_mode, yaml_file, first_self, last_self, name_method, dir_method, number_iteration):
    logger.info("starting self_fit")
    logger.info(f"params: {[feature_mode, yaml_file, first_self, last_self]}")

    FeatureReader.reader_selection_parameter = feature_mode
    csv_dir = DATA_PATH_INTERACTIONS / "train" / name_method / number_iteration
    files = list(csv_dir.glob('**/*.csv'))
    for f in files:
        results_dir = ROOT_PATH / "Results/models" / dir_method / number_iteration
        logger.info(f"results_dir = {results_dir}")
        logger.info(f"start dataset = {f}")
        worker(f, results_dir=results_dir, yaml_file=yaml_file, number_iteration=number_iteration)
        logger.info(f"finish dataset = {f}")
    logger.info("finish self_fit")


def build_classifiers_isolation_forest(number_iteration):
    number_iteration = str(number_iteration)
    yaml_file = "/sise/home/efrco/efrco-master/Classifier/yaml/one_class_params.yml"
    FeatureReader.reader_selection_parameter = "without_hot_encoding"
    self_fit("without_hot_encoding", yaml_file, 1, 2, name_method="one_class_svm", dir_method="models_isolation_forest", number_iteration=number_iteration)

    print("END main_primary")

# build_classifiers_isolation_forest(number_iteration=0)

from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

def isolation():
    FeatureReader.reader_selection_parameter = "without_hot_encoding"

    train = "/sise/home/efrco/efrco-master/data/train/underSampling/0/tarBase_human_negative_features_train_underSampling_method_0.csv"
    test = "/sise/home/efrco/efrco-master/data/test/one_class_svm/0/tarBase_human_negative_test_one_class.csv"
    # test = "/sise/home/efrco/efrco-master/data/test/underSampling/0/tarBase_human_negative_features_test_underSampling_method_0.csv"
    feature_reader = get_reader()
    X_train, y_train = feature_reader.file_reader(train)
    y_train = pd.DataFrame(y_train, columns=['Label'])
    train = pd.concat([X_train, y_train], axis=1)
    train = train.drop(train[train['Label'] == 0].sample(frac=0.8, random_state=40).index)
    y = train['Label']
    # y = [0 if i==1 else 1 for i in y ]
    print(train['Label'].value_counts(normalize=True))
    train.drop(columns=["Label"], inplace=True)

    X_test, y_test = feature_reader.file_reader(test)
    # print(y_test.value_counts(normalize=True))


    # y_test = pd.DataFrame(y_test)
    # test = pd.concat([X_test, y_test], axis=1)
    # test = test.drop(test[test['Label'] == 0].sample(frac=0.8, random_state=40).index)
    # y = test['Label']
    # test.drop(columns=["Label"], inplace=True)

    if_model = RandomForestClassifier(n_estimators=100, random_state=0).fit(train,y)
    # Predict the anomalies
    if_prediction = if_model.predict(X_test)
    test_score = accuracy_score(y_test, if_prediction)
    print("TEST results" , test_score)
    auc = roc_auc_score(y_test, if_prediction)
    print("TEST results auc" , auc)


    # Change the anomalies' values to make it consistent with the true values
    # if_prediction = [0 if i==-1 else 1 for i in if_prediction]
    # Check the model performance
    print(classification_report(y_test, if_prediction))

# isolation()