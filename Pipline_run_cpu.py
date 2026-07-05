# from Prepare_Data.Pipline_positive_data import generate_positive_interaction, generate_negative_interaction
# from Representative_Graph.Probability_Matrix import run_generate_matrices
# from Prepare_Data.Featuers_Embedding_merge import generate_features_embedding_df_merge
# from Classifier.train_test_stratifiedKFold import StratifiedKFold_splits
# from Classifier.ClassifierWithGridSearch import train_models
# from Classifier.results_test import results_summary
from Representative_Graph.GCNN_embedding_StratifiedKFold import generate_embedding
import os
from Prepare_Data.Featuers_Embedding_merge import generate_features_embedding_df_merge
from Classifier.ClassifierWithGridSearch import train_models
from Classifier.results_test import results_summary
from Expreiments_Analysis import Find_Best_Conf_Improved
from pathlib import Path

# 1 step - generate the sub-optimal interaction
# generate_positive_interaction()
# generate_negative_interaction()
# check after that all the number interaction get features

# 2 step- generate matrix of probabillity
# 3 step - pay attention that we need to run that for negative and positive data
# run_generate_matrices()

# 4 step - split to train and test
# StratifiedKFold_splits()

# 5 step - get embedding for each fold
# Netural
# csv_dir = Path('/mnt/new_groups/vaksler_group/Efrat/Results/experiment_runs_featuers_new_try/run_epochs_80.0_lr_0.006_din_128.0_dout_64.0_bs_64.0/Data_embedding/Netural')
# generate_embedding(26, save_embedding_dir=csv_dir)
#Improved
# csv_dir = Path('/mnt/new_groups/vaksler_group/Efrat/Results/experiment_runs_featuers_new_try/run_epochs_80.0_lr_0.006_din_128.0_dout_64.0_bs_64.0/Data_embedding/Improved')
# generate_embedding_improved(26, save_embedding_dir=csv_dir)
# csv_dir = Path('/mnt/new_groups/vaksler_group/Efrat/Results/experiment_runs_featuers_new_try/run_epochs_80.0_lr_0.006_din_128.0_dout_64.0_bs_64.0/Data_embedding/Gat')
# generate_embedding_gat(26, save_embedding_dir=csv_dir)
#

# csv_dir = Path('/mnt/new_groups/vaksler_group/Efrat/Results/experiment_runs_featuers_new_try/run_epochs_80.0_lr_0.006_din_128.0_dout_64.0_bs_64.0/Data_embedding/3Layer')
# generate_embedding_3Layer(26, save_embedding_dir=csv_dir)


# 6 step - merge the features for the vector embedding for each interaction
# generate_features_embedding_df_merge()

# 7 step - run model
# train_models(model_mod = "models_xgbs_combine_embedding", feature_mode= "with_embedding_vector")
# train_models(model_mod = "models_xgbs_without_embedding", feature_mode= "without_embedding_vector")

# 8 step - run model
# results_summary(model_mode="models_xgbs_combine_embedding", reader_mode="with_embedding_vector" )
# results_summary(model_mode="models_xgbs_without_embedding", reader_mode="without_embedding_vector" )


# Find_Best_Conf_Improved.find_best_conf()




def run_experiment_Netural(exp_id):
    # print("Writing to:", ROOT_PATH_PHD_GOAL_ONE / "consts/GNN_consts.py")


    base_path = Path('/mnt/new_groups/vaksler_group/Efrat/Results/RelGraphConv/run_epochs_80.0_lr_0.001_din_128.0_dout_256.0_bs_64.0')

    data_embedding_path = base_path / "Data_embedding"
    print(data_embedding_path)
    results_path = base_path
    DATA_TRAIN_TEST_folds = base_path / "Data_merge_embedding_features"
    Results_with_embedding = base_path / "models_xgbs_combine_embedding"
    Results_without_embedding = base_path / "models_xgbs_without_embedding"
    Results_dir_csv = base_path / "Results"
    Results_only_embedding =  base_path / "models_xgbs_only_embedding"

    os.makedirs(data_embedding_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(DATA_TRAIN_TEST_folds, exist_ok=True)
    os.makedirs(Results_with_embedding, exist_ok=True)
    os.makedirs(Results_without_embedding, exist_ok=True)
    os.makedirs(Results_only_embedding, exist_ok=True)
    os.makedirs(Results_dir_csv, exist_ok=True)


    print("[INFO] Running embedding generation...")
    generate_embedding(exp_id, save_embedding_dir=data_embedding_path)

    print("[INFO] Running embedding+features vector...")
    generate_features_embedding_df_merge(embedding_base_path=data_embedding_path, output_base_path=DATA_TRAIN_TEST_folds)

    print("[INFO] Training XGBoost with embedding vector...")
    train_models(exp_id=exp_id, model_mod="models_xgbs_combine_embedding",
                 feature_mode="with_embedding_vector",
                 base_dir=base_path)

    print("[INFO] Training XGBoost without embedding vector...")
    train_models(exp_id=0, model_mod="models_xgbs_without_embedding",
                 feature_mode="without_embedding_vector",
                 base_dir=base_path)

    print("[INFO] Training XGBoost only embedding vector...")
    train_models(exp_id=exp_id, model_mod="models_xgbs_only_embedding",
                 feature_mode="only_embedding_vector",
                 base_dir=base_path)

    print("[INFO] Evaluating results...")
    results_summary(exp_id=exp_id, model_mode="models_xgbs_combine_embedding",
                    reader_mode="with_embedding_vector",
                    base_dir=results_path)

    results_summary(exp_id = 0, model_mode="models_xgbs_without_embedding",
                    reader_mode="without_embedding_vector",
                    base_dir=results_path)

    results_summary(exp_id=exp_id, model_mode="models_xgbs_only_embedding",
                    reader_mode="only_embedding_vector",
                    base_dir=results_path)
run_experiment_Netural(14)


