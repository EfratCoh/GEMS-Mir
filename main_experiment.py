import argparse
import os
from pathlib import Path
# import pandas as pd
# from consts.GNN_consts import *
from Prepare_Data.Featuers_Embedding_merge import generate_features_embedding_df_merge
from Classifier.ClassifierWithGridSearch import train_models
from Classifier.results_test import results_summary
from Representative_Graph.GCNN_embedding_StratifiedKFold import generate_embedding
import pandas as pd
import importlib
from consts.global_consts import ROOT_PATH_PHD_GOAL_ONE
import os
from pathlib import Path

def update_gcnn_consts(number_epoch, learning_rate, dim_vector_in, dim_vector_out, batch_size, n_classes, output_path):

    consts_content = f"""# Auto-generated GCNN constants
number_epoch = {number_epoch}
learning_rate = {learning_rate}
dim_vector_in = {dim_vector_in}
dim_vector_out = {dim_vector_out}
batch_size = {batch_size}
n_classes = {n_classes}
    """

    with open(output_path, "w") as f:
       f.write(consts_content)

    print(f"[INFO] Written experiment constants to {output_path}")


def run_experiment(args, exp_id, number_epoch, learning_rate, dim_vector_in, dim_vector_out, batch_size, n_classes):
    # print("Writing to:", ROOT_PATH_PHD_GOAL_ONE / "consts/GNN_consts.py")


    consts_dir = ROOT_PATH_PHD_GOAL_ONE / "consts_experiment_val_set"
    consts_dir.mkdir(exist_ok=True)

    const_filename = f"GCNN_CONST_{exp_id}.py"
    const_path = consts_dir / const_filename

    update_gcnn_consts(
        number_epoch, learning_rate,
        dim_vector_in, dim_vector_out,
        batch_size, n_classes,
        output_path=const_path
    )


    base_path = Path(args.exp_dir)

    data_embedding_path = base_path / "Data_embedding/"
    print(data_embedding_path)
    results_path = base_path
    DATA_TRAIN_TEST_folds = base_path / "Data_merge_embedding_features/"
    Results_with_embedding = base_path / "models_xgbs_combine_embedding/"
    Results_without_embedding = base_path / "models_xgbs_without_embedding/"
    Results_dir_csv = base_path / "Results/"
    Results_only_embedding =  base_path / "models_xgbs_only_embedding/"

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

def parse_args_from_env():
    return {
        "exp_id": int(os.environ["exp_id"]),
        "number_epoch": int(float(os.environ["GCNN_EPOCHS"])),
        "learning_rate": float(os.environ["GCNN_LR"]),
        "dim_vector_in": int(float(os.environ["GCNN_DIM_IN"])),
        "dim_vector_out": int(float(os.environ["GCNN_DIM_OUT"])),
        "batch_size": int(float(os.environ["GCNN_BATCH_SIZE"])),
        "n_classes": int(float(os.environ["GCNN_N_CLASSES"])),
        "exp_dir": os.environ["GCNN_EXP_DIR"]
    }

if __name__ == "__main__":

    args = parse_args_from_env()
    print(args)
    class Struct:
        pass
    args_obj = Struct()
    args_obj.exp_dir = args["exp_dir"]
    print(args_obj)
    #
    run_experiment(
        args=args_obj,
        exp_id=int(args["exp_id"]),
        number_epoch=args["number_epoch"],
        learning_rate=args["learning_rate"],
        dim_vector_in=args["dim_vector_in"],
        dim_vector_out=args["dim_vector_out"],
        batch_size=args["batch_size"],
        n_classes=args["n_classes"]
    )