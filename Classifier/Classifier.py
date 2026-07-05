from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import accuracy_score
import pickle
import pandas as pd
from pathlib import Path
import yaml
from consts.global_consts import ROOT_PATH_PHD_GOAL_ONE
import pickle
# import Classifier.FeatureReader as FeatureReader
# from Classifier.FeatureReader import get_reader
# from Classifier.ClfLogger import logger

import FeatureReader as FeatureReader
from FeatureReader import get_reader
from ClfLogger import logger
from xgboost import XGBClassifier


class ClassifierWithNestedCV:
    def __init__(self, dataset_file, result_dir, outer_folds=5, inner_folds=3):
        self.dataset_file = dataset_file
        self.dataset_name = self.extract_dataset_name()
        print(f"Handling dataset: {self.dataset_name}")
        self.load_dataset()
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(exist_ok=True, parents=True)
        self.create_clf_dict()
        self.outer_folds = outer_folds
        self.inner_folds = inner_folds

    def create_clf_dict(self):
        self.clf_dict = {"xgbs": XGBClassifier()}

    def load_dataset(self):
        directory = self.dataset_file.parent
        feature_reader = get_reader()
        X, y = feature_reader.file_reader(directory / f"{self.dataset_name}.csv")
        self.X = X
        self.y = y

    def train_one_conf(self, clf_name, conf, scoring="accuracy"):
        output_file = self.result_dir / f"{self.dataset_name}_{clf_name}_nested_cv_results.csv"

        # Define classifier and parameter grid
        clf = self.clf_dict[clf_name]
        parameters = conf['parameters']

        outer_cv = KFold(n_splits=self.outer_folds, shuffle=True, random_state=42)
        outer_results = []

        print(
            f"Starting Nested Cross-Validation with {self.outer_folds} outer folds and {self.inner_folds} inner folds.")

        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(self.X)):
            print(f"Outer fold {fold_idx + 1}/{self.outer_folds}")

            # Split data for outer fold
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]

            # Inner CV for hyperparameter tuning
            grid_obj = GridSearchCV(
                estimator=clf,
                param_grid=parameters,
                scoring=scoring,
                cv=self.inner_folds,
                n_jobs=-1,
                verbose=3
            )
            grid_obj.fit(X_train, y_train)

            # Evaluate on the outer test set
            best_clf = grid_obj.best_estimator_
            y_pred = best_clf.predict(X_test)
            test_score = accuracy_score(y_test, y_pred)

            print(f"Outer fold {fold_idx + 1} - Best params: {grid_obj.best_params_}, Test score: {test_score:.4f}")

            # Save results for this fold
            outer_results.append({
                "fold": fold_idx + 1,
                "best_params": grid_obj.best_params_,
                "test_score": test_score
            })

        # Save results to a CSV file
        results_df = pd.DataFrame(outer_results)
        results_df.to_csv(output_file, index=False)

        print(f"Nested CV results saved to {output_file}")

    def fit(self, yaml_path):
        with open(yaml_path, 'r') as stream:
            training_config = yaml.safe_load(stream)

        for clf_name, conf in training_config.items():
            if conf["run"]:
                self.train_one_conf(clf_name, conf, scoring="accuracy")


# Example Worker Function
def worker(dataset_files, results_dir, yaml_file):
    clf_nested_cv = ClassifierWithNestedCV(dataset_file=dataset_files, result_dir=results_dir)
    clf_nested_cv.fit(yaml_file)


# Example Main Function
def build_classifiers(number_epoch):
    yaml_file = "/sise/home/efrco/efrco-master/Classifier/yaml/xgbs_params.yml"

    FeatureReader.reader_selection_parameter = "with_embedding_vector"
    csv_dir = ROOT_PATH_PHD_GOAL_ONE / "Data_merge_embedding_features" / number_epoch
    files = list(csv_dir.glob('**/*.csv'))
    results_dir = ROOT_PATH_PHD_GOAL_ONE / "Results/models" / number_epoch
    worker(files, results_dir=results_dir, yaml_file=yaml_file)

    print("Finished Nested CV!")


# Build classifiers with nested CV
# build_classifiers(number_epoch=0)

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import accuracy_score
import pandas as pd
from pathlib import Path
import yaml
import os
# from xgboost import XGBClassifier
# from consts.global_consts import ROOT_PATH_PHD_GOAL_ONE
# import Classifier.FeatureReader as FeatureReader
# from Classifier.FeatureReader import get_reader
# from Classifier.ClfLogger import logger


# Classifier with Nested CV
class ClassifierWithNestedCV:
    def __init__(self,dir_target, number_epoch,  result_dir, outer_folds=5, inner_folds=3):
        self.dir_target = dir_target
        self.number_epoch = number_epoch
        self.load_dataset()
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(exist_ok=True, parents=True)
        self.outer_folds = outer_folds
        self.inner_folds = inner_folds
        self.create_clf_dict()

    def create_clf_dict(self):
        self.clf_dict = {"xgbs": XGBClassifier()}

    def load_dataset(self, merge_df):
        name_dataset = f"merge_df_{self.number_epoch}.csv"
        feature_reader = get_reader()
        X, y = feature_reader.file_reader(self.dir_target / name_dataset)
        self.X = X
        self.y = y

    def train_one_conf(self, clf_name, conf, scoring="accuracy"):
        output_file = self.result_dir / f"nested_cv_results_{clf_name}.csv"

        # Define classifier and parameter grid
        clf = self.clf_dict[clf_name]
        parameters = conf['parameters']


        # Outer CV for performance estimation
        outer_cv = KFold(n_splits=self.outer_folds, shuffle=True, random_state=42)
        outer_results = []

        print(f"Starting Nested CV with {self.outer_folds} outer folds and {self.inner_folds} inner folds.")
        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(self.X)):
            print(f"Outer fold {fold_idx + 1}/{self.outer_folds}")

            # Split outer fold data
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]

            # Inner CV for hyperparameter tuning
            grid_obj = GridSearchCV(
                estimator=clf,
                param_grid=parameters,
                scoring=scoring,
                cv=self.inner_folds,
                n_jobs=-1,
                verbose=3
            )
            grid_obj.fit(X_train, y_train)

            # Evaluate the best model on the outer test set
            best_clf = grid_obj.best_estimator_
            y_pred = best_clf.predict(X_test)
            test_score = accuracy_score(y_test, y_pred)

            print(f"Outer fold {fold_idx + 1} - Best params: {grid_obj.best_params_}, Test score: {test_score:.4f}")

            # Save results for this fold
            outer_results.append({
                "fold": fold_idx + 1,
                "best_params": grid_obj.best_params_,
                "test_score": test_score
            })

        # Save all fold results
        results_df = pd.DataFrame(outer_results)
        results_df.to_csv(output_file, index=False)
        print(f"Nested CV results saved to {output_file}")

    def fit(self, yaml_path):
        with open(yaml_path, 'r') as stream:
            training_config = yaml.safe_load(stream)

        for clf_name, conf in training_config.items():
            if conf["run"]:
                self.train_one_conf(clf_name, conf, scoring="accuracy")

# Merge Positive and Negative Datasets
def merge_datasets(csv_dir, dict_map_label_dataset, number_epoch):
    files = list(csv_dir.glob('**/*.csv'))
    merged_data = []

    for file in files:
        for dataset_name, label in dict_map_label_dataset.items():
            if dataset_name in file.stem:
                df = pd.read_csv(file)
                df['label'] = label  # Assign label
                merged_data.append(df)

    if merged_data:
        merged_df = pd.concat(merged_data, axis=0, ignore_index=True)
        print("Merged datasets with assigned labels.")

        # Save the merged DataFrame
        save_path = csv_dir / f"merge_df_{number_epoch}.csv"
        merged_df.to_csv(save_path, index=False)
        print(f"Merged DataFrame saved to: {save_path}")

        return merged_df
    else:
        raise ValueError("No matching datasets found in the directory!")


# Main Workflow
def build_classifiers(number_epoch):
    yaml_file = "/sise/home/efrco/efrco-master/Classifier/yaml/xgbs_params.yml"

    FeatureReader.reader_selection_parameter = "with_embedding_vector"
    dir_target = ROOT_PATH_PHD_GOAL_ONE / "Data_merge_embedding_features" / number_epoch
    results_dir = ROOT_PATH_PHD_GOAL_ONE / "Results/models" / number_epoch

    # Define dataset label mapping
    dict_map_label_dataset = {"darnell_human_dataset": 0, "NPS_darnell_human_dataset": 1}

    # Merge datasets and save to CSV
    merged_df = merge_datasets(dir_target, dict_map_label_dataset, number_epoch)

    # Run Nested CV
    clf_nested_cv = ClassifierWithNestedCV(dir_target, number_epoch, result_dir=results_dir)
    clf_nested_cv.fit(yaml_file)

    print("Finished Nested CV!")

# Execute the code
# build_classifiers(number_epoch=0)
