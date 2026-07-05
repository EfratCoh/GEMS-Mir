from consts.global_consts import NEGATIVE_DATA_PATH, POSITIVE_PATH_FEATUERS
from sklearn.model_selection import StratifiedKFold
from pathlib import Path


pos_dataset_darnell = POSITIVE_PATH_FEATUERS/ "darnell_human_ViennaDuplex_75nt_fragment_clean_features.csv"
neg_dataset_NPS_CLASH = NEGATIVE_DATA_PATH / "non_overlapping_sites/non_overlapping_sites_darnell_human_ViennaDuplex_75nt_fragment_clean_negative_features.csv"


POSITIVE_DATASETS_ARR = [pos_dataset_darnell]
NEGATIVE_DATASETS_ARR = [neg_dataset_NPS_CLASH]