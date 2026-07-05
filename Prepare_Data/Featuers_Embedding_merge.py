import sys
import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from consts.global_consts import DATA_PATH_INTERACTIONS, MERGE_DATA, Data_Embedding,DATA_TRAIN_TEST,FOLDS_PATH, ROOT_PATH_PHD_GOAL_ONE, NEGATIVE_DATA_PATH, POSITIVE_PATH_FEATUERS
from utils.utilsfile import read_csv, to_csv, clean_directory
import os
import numpy as np
import pandas as pd



# Input/output paths
# embedding_base_path = Data_Embedding
# output_base_path = DATA_TRAIN_TEST

def generate_features_embedding_df_merge(embedding_base_path, output_base_path):

    folds_base_path = FOLDS_PATH

    clean_directory(output_base_path)

    # Loop over folds (e.g., fold1, fold2, ...)
    for fold_name in sorted(os.listdir(embedding_base_path)):
        fold_embed_path = os.path.join(embedding_base_path, fold_name)
        fold_data_path = os.path.join(folds_base_path, fold_name)
        if not os.path.isdir(fold_embed_path) or not os.path.isdir(fold_data_path):
            continue

        # Extract number from 'foldX'
        fold_number = fold_name.replace("fold", "")

        # Updated filenames according to your format
        train_features_path = os.path.join(fold_data_path, f"fold{fold_number}_train.csv")
        test_features_path = os.path.join(fold_data_path, f"fold{fold_number}_test.csv")

        if not os.path.exists(train_features_path) or not os.path.exists(test_features_path):
            print(f"⚠️ Missing train/test files in: {fold_data_path}")
            continue

        train_features_df = pd.read_csv(train_features_path)
        test_features_df = pd.read_csv(test_features_path)

        # Loop over epochs (epoch_0, epoch_1, ...)
        for epoch_name in sorted(os.listdir(fold_embed_path)):
            epoch_path = os.path.join(fold_embed_path, epoch_name)
            if not os.path.isdir(epoch_path):
                continue

            # Embedding files
            epoch_train_embed = os.path.join(epoch_path, "train_embeddings.csv")
            epoch_test_embed = os.path.join(epoch_path, "test_embeddings.csv")

            if not os.path.exists(epoch_train_embed) or not os.path.exists(epoch_test_embed):
                print(f"⚠️ Missing embedding files in: {epoch_path}")
                continue

            try:
                # Load embeddings
                train_embed_df = pd.read_csv(epoch_train_embed)
                test_embed_df = pd.read_csv(epoch_test_embed)

                # Merge by ID_interaction
                merged_train = pd.merge(train_features_df, train_embed_df, on="ID_interaction", how="inner")
                merged_test = pd.merge(test_features_df, test_embed_df, on="ID_interaction", how="inner")

                # Create output directory
                output_dir = os.path.join(output_base_path, fold_name, epoch_name)
                os.makedirs(output_dir, exist_ok=True)

                # Save merged files
                epoch_number = epoch_name.replace("epoch_", "")

                merged_train.to_csv(os.path.join(output_dir, f"merged_train_fold_{fold_number}_epoch_{epoch_number}.csv"),
                                    index=False)
                merged_test.to_csv(os.path.join(output_dir, f"merged_test_fold_{fold_number}_epoch_{epoch_number}.csv"),
                                   index=False)

                print(f"✅ Merged and saved: Fold {fold_number}, Epoch {epoch_number}")

            except Exception as e:
                print(f"❌ Error merging in {fold_name}/{epoch_name}: {e}")
