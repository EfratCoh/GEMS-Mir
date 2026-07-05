from consts.datasets_path import POSITIVE_DATASETS_ARR, NEGATIVE_DATASETS_ARR
from consts.global_consts import FOLDS_PATH, number_fold, Data_Embedding, number_epoch, ROOT_PATH_PHD_GOAL_ONE, NEGATIVE_DATA_PATH, POSITIVE_PATH_FEATUERS, MATRIX_DATA
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
from utils.utilsfile import read_csv, to_csv
import pandas as pd

def StratifiedKFold_splits():
    # Load and label data
    pos_df_list = [read_csv(path).assign(label=1) for path in POSITIVE_DATASETS_ARR]
    neg_df_list = [read_csv(path).assign(label=0) for path in NEGATIVE_DATASETS_ARR]

    # Concatenate all positive and negative datasets
    pos_df = pd.concat(pos_df_list, ignore_index=True)
    neg_df = pd.concat(neg_df_list, ignore_index=True)

    # Combine and shuffle
    all_data = pd.concat([pos_df, neg_df], ignore_index=True)
    all_data = all_data.sample(frac=1, random_state=42).reset_index(drop=True)

    # Prepare cross-validation
    X = all_data.drop(columns=['label'])
    y = all_data['label']
    skf = StratifiedKFold(n_splits=number_fold, shuffle=True, random_state=42)

    # Iterate over folds
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"Processing Fold {fold+1}/10")

        train_df = all_data.iloc[train_idx].reset_index(drop=True)
        test_df = all_data.iloc[test_idx].reset_index(drop=True)

        fold_dir = FOLDS_PATH / f"fold{fold + 1}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        # Save train/test CSVs
        # train_df.drop(labels=['level_0'], axis=1, inplace=True)
        # test_df.drop(labels=['level_0'], axis=1, inplace=True)

        to_csv(train_df, fold_dir / f"fold{fold + 1}_train.csv")
        to_csv(test_df, fold_dir / f"fold{fold + 1}_test.csv")


# StratifiedKFold_splits()

