import itertools
import subprocess
from pathlib import Path
import pandas as pd
import json
import os
from consts.global_consts import ROOT_PATH_PHD_GOAL_ONE

# Define grid of parameters
# params = {
#     "number_epoch": [80, 40, 50, 100, 120],
#     "learning_rate": [0.001, 0.006],
#     "dim_vector_in": [64, 128],
#     "dim_vector_out": [64, 128, 256, 300],
#     "batch_size": [32, 64],
#     "n_classes": [2]
# }
params = {
    "number_epoch": [80],
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

# Iterate over each experiment
# for i, row in param_df.iterrows():
#     folder_name = f"run_epochs_{row['number_epoch']}_lr_{row['learning_rate']}_din_{row['dim_vector_in']}_dout_{row['dim_vector_out']}_bs_{row['batch_size']}"
#     exp_path = base_dir / folder_name
#     exp_path.mkdir(parents=True, exist_ok=True)
#
#     print(f"[RUNNING] Experiment {i+1}/{len(param_df)}: {folder_name}")

main_script = Path("/home/efrco/PHD/Goal_One/main_experiment.py")

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
    python_path = "/home/efrco/.local/share/virtualenvs/efrco-master-BYjOffaj/bin/python3.7"

    result = subprocess.run([
        python_path, str(main_script)
    ], env=env_vars, capture_output=True, text=True)

    # print("Return code:", result.returncode)
    # print("STDOUT:\n", result.stdout)
    # print("STDERR:\n", result.stderr)
    print(f"[FINISH] Experiment {i + 1}/{len(param_df)}: {folder_name}")

    # # Save stdout to log file
    # with open(exp_path / "experiment_log.txt", "w") as log_file:
    #     log_file.write(result.stdout)
    #     log_file.write("\nSTDERR:\n")
    #     log_file.write(result.stderr)
    #
    # # Try to find results.csv in exp_path / Results directory
    # results_csv_path = exp_path / "Results" / "summary_metrics.csv"
    # if results_csv_path.exists():
    #     df_result = pd.read_csv(results_csv_path)
    #     df_result["experiment_folder"] = folder_name
    #     for param in row.index:
    #         df_result[param] = row[param]
    #     results_summary_list.append(df_result)

# # Save all combined results
# if results_summary_list:
#     final_df = pd.concat(results_summary_list, ignore_index=True)
#     final_df.to_csv(base_dir / "all_experiment_results.csv", index=False)
#     print(f"[SUCCESS] All experiment results saved to {base_dir / 'all_experiment_results.csv'}")
# else:
#     print("[WARNING] No results found to summarize.")
