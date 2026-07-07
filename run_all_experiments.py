import itertools
import subprocess
from pathlib import Path
import pandas as pd
import json
import os
from consts.global_consts import ROOT_PATH_PHD_GOAL_ONE

# Define grid of parameters
params = {
    "number_epoch": [80, 40, 50, 100, 120],
    "learning_rate": [0.001, 0.006],
    "dim_vector_in": [64, 128],
    "dim_vector_out": [64, 128, 256, 300],
    "batch_size": [32, 64],
    "n_classes": [2]
}


param_combinations = list(itertools.product(*params.values()))
param_df = pd.DataFrame(param_combinations, columns=params.keys())

# Base directory to store experiments
base_dir = Path("/groups/vaksler_group/Efrat/Results/experiment_runs_featuers_val_set/")
base_dir.mkdir(parents=True, exist_ok=True)

# Save parameters for reference
param_df.to_csv(base_dir / "experiment_grid.csv", index=False)

# Initialize list to collect results
results_summary_list = []

main_script = Path("./main_experiment.py")

# Iterate over each experiment
for i, row in param_df.iterrows():

    folder_name = f"run_epochs_{row['number_epoch']}_lr_{row['learning_rate']}_din_{row['dim_vector_in']}_dout_{row['dim_vector_out']}_bs_{row['batch_size']}"
    exp_path = base_dir / folder_name
    # exp_path.mkdir(parents=True, exist_ok=True)
    if exp_path.exists():
        print("The directory already exists, proceeding...")
        continue
    else:
        exp_path.mkdir(parents=True)
        print("The directory has been created.")


    print(f"[RUNNING] Experiment {i + 1}/{len(param_df)}: {folder_name}")

    env_vars = {
        **os.environ,
        "exp_id" : str(i+1),
        "GCNN_EPOCHS": str(row['number_epoch']),
        "GCNN_LR": str(row['learning_rate']),
        "GCNN_DIM_IN": str(row['dim_vector_in']),
        "GCNN_DIM_OUT": str(row['dim_vector_out']),
        "GCNN_BATCH_SIZE": str(row['batch_size']),
        "GCNN_N_CLASSES": str(row['n_classes']),
        "GCNN_EXP_DIR": str(exp_path)
    }
    python_path = "./bin/python3.7"

    result = subprocess.run([
        python_path, str(main_script)
    ], env=env_vars, capture_output=True, text=True)


    print(f"[FINISH] Experiment {i + 1}/{len(param_df)}: {folder_name}")

   
