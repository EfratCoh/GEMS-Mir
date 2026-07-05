from pathlib import Path
import yaml
from consts.global_consts import ROOT_PATH_PHD_GOAL_ONE,DATA_TRAIN_TEST,number_fold, MERGE_DATA, DATA_PATH_INTERACTIONS
from xgboost import XGBClassifier
import pickle
import Classifier.FeatureReader as FeatureReader
from Classifier.FeatureReader import get_reader
from Classifier.ClfLogger import logger
# from utils.utilsfile import read_csv, to_csv, clean_directory, get_wrapper, import_consts
from sklearn.metrics import accuracy_score
# import FeatureReader as FeatureReader
# from FeatureReader import get_reader
# from ClfLogger import logger
from utils.utilsfile import read_csv, to_csv, import_consts
from pathlib import Path
import re


class ClassifierWithGridSearch(object):
    def __init__(self, dataset_file, result_dir, exp_id):
        self.dataset_file = dataset_file
        self.dataset_name = self.extract_dataset_name()
        print(f"Handling dataset : {self.dataset_name}")
        self.exp_id = exp_id
        self.load_dataset()
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(exist_ok=True, parents=True)
        self.create_clf_dict()

    def extract_dataset_name(self):
        return str(self.dataset_file.stem).split("_test")[0]

    def create_clf_dict(self):
        self.clf_dict = {
            "xgbs": XGBClassifier(use_label_encode=False),
            "xgbs_no_encoding": XGBClassifier(),
        }

    # this function response on load the dataset
    def load_dataset(self):
        directory = self.dataset_file.parent
        feature_reader = get_reader()
        feature_reader.setnumbrtexperiments(self.exp_id)
        X, y = feature_reader.file_reader(directory / f"{self.dataset_name}.csv")
        self.X = X
        self.y = y


    # this function response on train model and then save this model
    def train_one_conf(self, clf_name, conf, scoring="accuracy"):

        output_file = self.result_dir / f"{self.dataset_name.split('merged_train_')[1]}_{clf_name}.csv"

        # creat the specific clf and load the parameters of the clf according to the ymal file.

        model_params = conf["model_parameters"]
        fit_params = conf["fit_parameters"]

        grid_obj = XGBClassifier(**model_params)
        grid_obj.fit(self.X, self.y, **fit_params)

        print("\n Final trained model:")
        print(grid_obj)
        y_pred = grid_obj.predict(self.X)

        train_accuracy = accuracy_score(self.y, y_pred)

        print(f"Train accuracy: {train_accuracy:.4f}")

        model_file = self.result_dir / f"{self.dataset_name.split('merged_train_')[1]}_{clf_name}.model"

        try:
            with model_file.open("wb") as pfile:
                pickle.dump(grid_obj, pfile)
        except Exception:
            pass

    def fit(self, yaml_path):
        with open(yaml_path, 'r') as stream:
            training_config = yaml.safe_load(stream)

        for clf_name, conf in training_config.items():
            if conf["run"]:
                self.train_one_conf(clf_name, conf, scoring="accuracy")
                #clf = self.fit_best_clf(clf_name)
                #print(clf)


def worker(dataset_file, results_dir, yaml_file, exp_id):
    clf_grid_search = ClassifierWithGridSearch(dataset_file=dataset_file, result_dir=results_dir, exp_id=exp_id)
    clf_grid_search.fit(yaml_file)
    return


def self_fit(feature_mode,models_dir, yaml_file, number_fold, number_epoch, base_dir, exp_id):
    logger.info("starting self_fit")
    logger.info(f"params: {[feature_mode, yaml_file]}")

    FeatureReader.reader_selection_parameter = feature_mode
    csv_dir = base_dir / "Data_merge_embedding_features" / f'fold{number_fold}' / f'epoch_{number_epoch}'
    files = list(csv_dir.glob('**/*.csv'))

    for f in sorted(files):
        if not f.name.startswith("merged_train_"):
            continue
        results_dir = base_dir / f'{models_dir}' / f'fold{number_fold}'
        results_dir.mkdir(parents=True, exist_ok=True)
        # if int(number_epoch)==0:
        #     clean_directory(results_dir)

        logger.info(f"results_dir = {results_dir}")
        logger.info(f"start dataset = {f}")
        worker(f, results_dir=results_dir, yaml_file=yaml_file, exp_id=exp_id)
        logger.info(f"finish dataset = {f}")
    logger.info("finish self_fit")


def build_classifiers(number_fold, number_epoch, models_dir,feature_mode, base_dir, exp_id):
    # yaml_file = "/sise/home/efrco/PHD/Goal_One/Classifier/yaml/xgbs_params_small.yml"
    # yaml_file = "/sise/home/efrco/PHD/Goal_One/Classifier/yaml/xgbs_params.yml"
    yaml_file = "/home/efrco/PHD/Goal_One/Classifier/yaml/xgbs_params_default.yml"
    number_fold = str(number_fold)
    number_epoch = str(number_epoch)

    self_fit(feature_mode=feature_mode,models_dir=models_dir, yaml_file=yaml_file, number_fold= number_fold, number_epoch=number_epoch, base_dir=base_dir, exp_id=exp_id)
    print("END main_primary")


def count_epochs(base_dir, fold):
    fold_path = Path(base_dir) / "Data_embedding" / f"fold{fold}"
    epochs = sorted(int(p.name.split("_")[1]) for p in fold_path.glob("epoch_*"))
    return len(epochs)

def train_models(exp_id, model_mod, feature_mode, base_dir):
    for fold in range(1, number_fold+1):
        number_epoch = count_epochs(base_dir, fold)
        for epoch in range(0,number_epoch):
            build_classifiers(number_fold=fold, number_epoch=epoch,models_dir = model_mod,  feature_mode=feature_mode, base_dir=base_dir, exp_id = exp_id)
#
# base_dir = Path("/mnt/new_groups/vaksler_group/Efrat/Results/experiment_runs_16_featuers/run_epochs_80.0_lr_0.006_din_128.0_dout_256.0_bs_64.0/")
# train_models(exp_id= 1, model_mod = "models_xgbs_combine_embedding", feature_mode= "with_embedding_vector", base_dir =base_dir)
# train_models(exp_id= 0, model_mod = "models_xgbs_without_embedding", feature_mode= "without_embedding_vector", base_dir= base_dir)