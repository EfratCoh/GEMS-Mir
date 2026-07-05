from sklearn.metrics import roc_auc_score, roc_curve
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
import pickle
import Classifier.FeatureReader as FeatureReader
from Classifier.FeatureReader import get_reader
from scipy.stats import spearmanr
import shap

# ======================================================
#               CONFIGURATION
# ======================================================

# Significant improve
# base_dir = "/groups/vaksler_group/Efrat/Results/experiment_runs_16_featuers/run_epochs_80.0_lr_0.006_din_128.0_dout_256.0_bs_64.0/"

#new tring
base_dir = Path("/mnt/new_groups/vaksler_group/Efrat/Results/experiment_runs_featuers_new_try/run_epochs_100.0_lr_0.001_din_128.0_dout_300.0_bs_64.0/")
base_dir = Path("/mnt/new_groups/vaksler_group/Efrat/Results/experiment_runs_featuers_new_try/run_epochs_100.0_lr_0.001_din_64.0_dout_300.0_bs_32.0/")
base_dir = Path("/mnt/new_groups/vaksler_group/Efrat/Results/experiment_runs_featuers_without_backbone/run_epochs_80.0_lr_0.001_din_64.0_dout_128.0_bs_64.0/")
baseline_model = "models_xgbs_without_embedding"
improved_model = "models_xgbs_combine_embedding"
baseline_reader = "without_embedding_vector"
improved_reader = "with_embedding_vector"
data_root_name = "Data_merge_embedding_features"


# ======================================================
#               DELONG FUNCTIONS
# ======================================================

def roc_variance(ground_truth, predictions):
    ground_truth = np.array(ground_truth)
    predictions = np.array(predictions)

    pos = predictions[ground_truth == 1]
    neg = predictions[ground_truth == 0]

    m = len(pos)
    n = len(neg)

    U = sum(np.sum(p > neg) + 0.5*np.sum(p == neg) for p in pos)
    auc_val = U / (m * n)

    V10 = np.array([(np.sum(pos[i] > neg) + 0.5*np.sum(pos[i] == neg)) for i in range(m)]) / n
    V01 = np.array([(np.sum(pos > neg[j]) + 0.5*np.sum(pos == neg[j])) for j in range(n)]) / m

    var_auc = np.var(V10)/m + np.var(V01)/n
    return auc_val, var_auc


def delong_roc_test(y_true, pred1, pred2):
    auc1, var1 = roc_variance(y_true, pred1)
    auc2, var2 = roc_variance(y_true, pred2)
    se = np.sqrt(var1 + var2)
    z = (auc1 - auc2) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return auc1, auc2, p_value


# ======================================================
#            STEP 1: COLLECT PREDICTIONS
# ======================================================

def collect_predictions(base_dir, model_mode, reader_mode):
    FeatureReader.reader_selection_parameter = reader_mode
    feature_reader = get_reader()
    if reader_mode == baseline_reader:
        feature_reader.setnumbrtexperiments(exp_id = 0)
    if reader_mode == improved_reader:
        feature_reader.setnumbrtexperiments(exp_id=1)

    predictions = []
    model_root = Path(base_dir) / model_mode
    data_root = Path(base_dir) / data_root_name

    for fold_path in sorted(model_root.glob("fold*")):
        fold = fold_path.name.replace("fold", "")

        for model_file in sorted(fold_path.glob("*.model")):
            epoch = model_file.stem.split("_")[3]

            print(f"[RUN] Fold {fold} Epoch {epoch} — {model_mode}")

            with open(model_file, "rb") as f:
                model = pickle.load(f)

            test_csv = data_root / f"fold{fold}" / f"epoch_{epoch}" / f"merged_test_fold_{fold}_epoch_{epoch}.csv"
            if not test_csv.exists():
                continue

            X_test, y_test = feature_reader.file_reader(test_csv)
            y_prob = model.predict_proba(X_test)[:, 1]

            predictions.append({
                "fold": int(fold),
                "epoch": int(epoch),
                "y_true": y_test,
                "y_prob": y_prob
            })

            print(f"[OK] Fold {fold} Epoch {epoch} — {model_mode}: AUC={roc_auc_score(y_test, y_prob):.4f}")

    return predictions


print("\n===== STARTING PREDICTION COLLECTION =====")

baseline_predictions = collect_predictions(base_dir, baseline_model, baseline_reader)
improved_predictions = collect_predictions(base_dir, improved_model, improved_reader)

df_base = pd.DataFrame(baseline_predictions)
df_improved = pd.DataFrame(improved_predictions)

df_base.to_pickle(Path(base_dir) / "baseline_predictions.pkl")
df_improved.to_pickle(Path(base_dir) / "improved_predictions.pkl")

print("Saved baseline + improved prediction files.\n")


# ======================================================
# STEP 2: CREATE DELONG RESULTS PER FOLD/EPOCH
# ======================================================
# Load predictions created in STEP 1
df_base = pd.read_pickle(Path(base_dir) / "baseline_predictions.pkl")
df_improved = pd.read_pickle(Path(base_dir) / "improved_predictions.pkl")
print("[INFO] Loaded baseline_predictions.pkl and improved_predictions.pkl")


merged = pd.merge(df_base, df_improved, on=["fold","epoch"], suffixes=("_base","_improved"))

pvals = []
auc_base_list = []
auc_improve_list = []

for _, row in merged.iterrows():
    y_true = row["y_true_base"]
    pb = row["y_prob_base"]
    pi = row["y_prob_improved"]

    auc_b = roc_auc_score(y_true, pb)
    auc_i = roc_auc_score(y_true, pi)

    auc_base_list.append(auc_b)
    auc_improve_list.append(auc_i)

    pval = delong_roc_test(y_true, pb, pi)[2]
    pvals.append(pval)

merged["auc_base"] = auc_base_list
merged["auc_improved"] = auc_improve_list
merged["p_value"] = pvals
merged["delta_auc"] = merged["auc_improved"] - merged["auc_base"]

merged.to_csv(Path(base_dir) / "delong_results.csv", index=False)

print("Saved delong_results.csv\n")


# ======================================================
# STEP 3: OPTION B → Best Epoch by Average ΔAUC
# ======================================================

merged = pd.read_csv(Path(base_dir) / "delong_results.csv")
print("[INFO] Loaded delong_results.csv for Step 3")


epoch_stats = merged.groupby("epoch").agg(
    mean_auc_base=("auc_base", "mean"),
    mean_auc_improved=("auc_improved", "mean"),
    mean_delta=("delta_auc", "mean"),
).reset_index()

best_epoch = epoch_stats.loc[epoch_stats["mean_delta"].idxmax(), "epoch"]
print("===== BEST EPOCH SELECTED =====")
print(epoch_stats)
print(f"\nBest epoch based on ΔAUC mean: {best_epoch}\n")


# ======================================================
# STEP 4: POOLED DELONG TEST
# ======================================================

dfb = df_base[df_base["epoch"] == best_epoch]
dfi = df_improved[df_improved["epoch"] == best_epoch]

pooled = pd.merge(dfb, dfi, on="fold", suffixes=("_base","_improved"))

y_true_pool = np.concatenate(pooled["y_true_base"].values)
pb_pool = np.concatenate(pooled["y_prob_base"].values)
pi_pool = np.concatenate(pooled["y_prob_improved"].values)


rho, _ = spearmanr(pb_pool, pi_pool)
print("Spearman correlation:", rho)

auc_b, auc_i, p_val = delong_roc_test(y_true_pool, pb_pool, pi_pool)

print("===== FINAL POOLED DELONG RESULT =====")
print(f"Baseline AUC:    {auc_b:.4f}")
print(f"Improved AUC:    {auc_i:.4f}")
print(f"ΔAUC:            {auc_i - auc_b:.4f}")
print(f"p-value:         {p_val:.6f}")

if p_val < 0.05:
    print("✔ Improvement IS statistically significant.")
else:
    print("❌ Improvement is NOT statistically significant.")

summary = pd.DataFrame([{
    "best_epoch": best_epoch,
    "auc_base": auc_b,
    "auc_improved": auc_i,
    "delta_auc": auc_i - auc_b,
    "p_value": p_val
}])

summary.to_csv(Path(base_dir) / "pooled_delong_summary.csv", index=False)
print("\nSaved pooled_delong_summary.csv\n")

# ======================================================
# STEP 4: Per-Fold Significance Summary
# ======================================================

# Load DeLong results
df = pd.read_csv(Path(base_dir) / "delong_results.csv")

# Filter only the selected best epoch
df_best_epoch = df[df["epoch"] == best_epoch]

significance_level = 0.05

num_folds = df_best_epoch.shape[0]
num_significant = (df_best_epoch["p_value"] < significance_level).sum()
percent_significant = 100 * num_significant / num_folds

mean_p_value = df_best_epoch["p_value"].mean()
median_p_value = df_best_epoch["p_value"].median()

print("\n===== PER-FOLD SIGNIFICANCE (BEST EPOCH ONLY) =====")
print(f"Epoch evaluated: {best_epoch}")
print(f"Mean p-value:    {mean_p_value:.4e}")
print(f"Median p-value:  {median_p_value:.4e}")
print(f"Significant folds (p < 0.05): {num_significant}/{num_folds} "
      f"({percent_significant:.1f}%)")

if percent_significant == 100:
    print("✔ Improvement is statistically significant in ALL folds.")
elif percent_significant >= 70:
    print("✔ Improvement is statistically significant in MOST folds.")
else:
    print("⚠ Improvement is NOT consistent across folds.")

# ======================================================
# STEP 5: PLOTS
# ======================================================

# Plot 1: Mean ΔAUC per epoch
plt.figure(figsize=(8,5))
plt.plot(epoch_stats["epoch"], epoch_stats["mean_delta"], marker="o")
plt.axhline(0, linestyle="--", color="black")
plt.title("Mean ΔAUC per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Mean ΔAUC")
plt.grid(True)
plt.savefig(Path(base_dir) / "mean_delta_auc_per_epoch.png")
plt.close()

# Plot 2: Pooled ROC curves
fpr_b, tpr_b, _ = roc_curve(y_true_pool, pb_pool)
fpr_i, tpr_i, _ = roc_curve(y_true_pool, pi_pool)

plt.figure(figsize=(8,6))
plt.plot(fpr_b, tpr_b, label=f"Baseline AUC={auc_b:.3f}")
plt.plot(fpr_i, tpr_i, label=f"Improved AUC={auc_i:.3f}")
plt.plot([0,1],[0,1],"--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"Pooled ROC Curve (Epoch {best_epoch})")
plt.legend()
plt.grid(True)
plt.savefig(Path(base_dir) / "pooled_roc_curve.png")
plt.close()

# Plot 3: Histogram ΔAUC for best epoch
best_epoch_rows = merged[merged["epoch"] == best_epoch]

plt.figure(figsize=(8,5))
plt.hist(best_epoch_rows["delta_auc"], bins=8, edgecolor="black")
plt.title(f"ΔAUC Distribution Across Folds (Epoch {best_epoch})")
plt.xlabel("ΔAUC")
plt.ylabel("Count")
plt.grid(True)
plt.savefig(Path(base_dir) / "delta_auc_histogram.png")
plt.close()

# ======================================================
#   PLOT: Baseline vs Improved AUC per Epoch
# ======================================================

epoch_auc = merged.groupby("epoch").agg(
    mean_auc_base=("auc_base", "mean"),
    mean_auc_improved=("auc_improved", "mean")
).reset_index()

plt.figure(figsize=(9,6))
plt.plot(epoch_auc["epoch"], epoch_auc["mean_auc_base"], marker="o", label="Baseline AUC")
plt.plot(epoch_auc["epoch"], epoch_auc["mean_auc_improved"], marker="o", label="Improved AUC")

plt.title("Baseline vs Improved AUC per Epoch")
plt.xlabel("Epoch")
plt.ylabel("AUC")
plt.grid(True)
plt.legend()
plt.savefig(Path(base_dir) / "baseline_vs_improved_auc_per_epoch.png")
plt.close()

print("Saved: baseline_vs_improved_auc_per_epoch.png")


print("Saved all plots!")
print(" - mean_delta_auc_per_epoch.png")
print(" - pooled_roc_curve.png")
print(" - delta_auc_histogram.png")


# ======================================================
# STEP 6: SHAP ANALYSIS (Improved Model - Test Set)
# ======================================================


print("\n===== STARTING SHAP ANALYSIS (Improved Model) =====")

# ------------------------------------------------------
# Load improved model for best epoch (one fold is enough)
# ------------------------------------------------------

example_fold = dfi.iloc[0]["fold"]
basline_epoch = 0
arr_epoch = [best_epoch, basline_epoch]
for example_epoch in arr_epoch:
    model_path = (
        Path(base_dir)
        / improved_model
        / f"fold{example_fold}"
    )

    model_file = [p for p in model_path.glob("*.model") if f"_{example_epoch}_" in p.name][0]

    with open(model_file, "rb") as f:
        shap_model = pickle.load(f)

    print(f"[INFO] Loaded model for SHAP: {model_file.name}")

    # ------------------------------------------------------
    # Load TEST data for SHAP (same fold & epoch)
    # ------------------------------------------------------

    FeatureReader.reader_selection_parameter = improved_reader
    feature_reader = get_reader()

    test_csv = (
        Path(base_dir)
        / data_root_name
        / f"fold{example_fold}"
        / f"epoch_{example_epoch}"
        / f"merged_test_fold_{example_fold}_epoch_{example_epoch}.csv"
    )

    X_test, y_test = feature_reader.file_reader(test_csv)

    print(f"[INFO] Loaded test data for SHAP: {test_csv.name}")
    print("X_test shape:", X_test.shape)

    # ------------------------------------------------------
    # SHAP Explainer (TreeExplainer – perfect for XGBoost)
    # ------------------------------------------------------

    explainer = shap.TreeExplainer(shap_model)

    shap_values = explainer.shap_values(X_test)

    print("[INFO] SHAP values computed")

    # ------------------------------------------------------
    # Global explanation plots
    # ------------------------------------------------------

    # Summary plot (most important)
    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=X_test.columns,
        title = f"shap_{example_fold}_epoch_{example_epoch}",
        show=False
    )
    plt.title(f"SHAP Summary – Fold {example_fold}, Epoch {example_epoch}")
    plt.tight_layout()
    plt.savefig(Path(base_dir) /f"shap_{example_fold}_epoch_{example_epoch}.png", bbox_inches="tight")
    plt.close()

    # Bar plot (global importance)
    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=X_test.columns,
        plot_type="bar",
        show=False
    )

    plt.savefig(Path(base_dir) / "shap_bar_plot.png", bbox_inches="tight")
    plt.close()

    print("Saved SHAP plots:")
    print(" - shap_summary_plot.png")
    print(" - shap_bar_plot.png")

print("\n===== SHAP ANALYSIS COMPLETED =====\n")


significance_level = 0.05
min_significant_folds = 6

df = pd.read_csv(Path(base_dir) / "delong_results.csv")

epoch_summary = (
    df.groupby("epoch")
      .agg(
          mean_auc_base=("auc_base", "mean"),
          mean_auc_improved=("auc_improved", "mean"),
          mean_delta=("delta_auc", "mean"),
          num_significant=("p_value", lambda x: (x < significance_level).sum()),
          num_folds=("p_value", "count")
      )
      .reset_index()
)

epoch_summary["percent_significant"] = (
    100 * epoch_summary["num_significant"] / epoch_summary["num_folds"]
)

print("\n===== EPOCH SUMMARY =====")
print(epoch_summary.sort_values("mean_delta", ascending=False).head(10))

balanced_epochs = epoch_summary[
    epoch_summary["num_significant"] >= min_significant_folds
].sort_values("mean_delta", ascending=False)

print("\n===== BALANCED EPOCH CANDIDATES =====")

if balanced_epochs.empty:
    print("❌ No epoch satisfies the consistency criterion.")
else:
    print(balanced_epochs[[
        "epoch",
        "mean_delta",
        "num_significant",
        "percent_significant"
    ]].head(10))

    best_balanced_epoch = balanced_epochs.iloc[0]["epoch"]

    print("\n===== BEST BALANCED EPOCH SELECTED =====")
    print(f"Epoch: {best_balanced_epoch}")




print("\n===== FULL END-TO-END ANALYSIS COMPLETED =====\n")


########3