from consts.global_consts import ROOT_PATH_PHD, MERGE_DATA
from pathlib import Path
from pandas import DataFrame, Series
import pandas as pd
from BreathesDuplex.Duplex_step import duplex as duplex_positive
from utils.logger import logger
from utils.utilsfile import get_wrapper, read_csv, to_csv

# This code generate for each interaction all possibility sub-optimal interactions
# The function duplex_positive responsible for generate the sub-optimal and save for each interaction file
# that contain the sub-optimal by the ID and the name of the dataset.

pos_dir_name = MERGE_DATA / "positive_interactions_new/data_without_featuers/"
for dataset_file in pos_dir_name.glob("*.csv"):
    dataset_name = str(dataset_file.stem).split("_features.csv")[0].split("_features")[0].split("_ViennaDuplex")[0]
    print(dataset_name)
    data_set_dir_name= dataset_name + "_dataset"
    path = ROOT_PATH_PHD / "Data_Breath_Duplex/"
    path_dir_target = path / data_set_dir_name
    current_positive_data = read_csv(dataset_file)
    # current_positive_data['ID_interaction'] = [f"{dataset_name}_{i}" for i in range(len(current_positive_data))]
    # to_csv(current_positive_data,dataset_file)
    print("###############Duplex POSITIVE#############")
    duplex_positive('ViennaDuplex', dataset_file, path_dir_target)
    break



