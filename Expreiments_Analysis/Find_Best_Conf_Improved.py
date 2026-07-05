import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve
from scipy import stats
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import shap
import seaborn as sns
sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "figure.dpi": 140
})

import Classifier.FeatureReader as FeatureReader
from Classifier.FeatureReader import get_reader


# ============================================
#            GLOBAL CONFIG
# ============================================

EXPERIMENTS_ROOT = Path(
    "/mnt/new_groups/vaksler_group/Efrat/Results/experiment_runs_featuers_without_backbone"
)
# EXPERIMENTS_ROOT = Path("/mnt/new_groups/vaksler_group/Efrat/Results/experiment_runs_featuers_new_try/run_epochs_80.0_lr_0.006_din_128.0_dout_64.0_bs_64.0/3LayerImproved/")
# EXPERIMENTS_ROOT = Path("/mnt/new_groups/vaksler_group/Efrat/Results/experiment_runs_featuers_new_try/run_epochs_80.0_lr_0.006_din_128.0_dout_64.0_bs_64.0/3LayerImproved/")

# Model directories (inside each run_epochs_* configuration directory)
BASELINE_MODEL_DIR_NAME = "models_xgbs_without_embedding"
IMPROVED_MODEL_DIR_NAME = "models_xgbs_combine_embedding"
ONLY_EMBEDDING_MODEL_DIR_NAME = "models_xgbs_only_embedding"

# Reader modes (FeatureReader selector)
BASELINE_READER_MODE = "without_embedding_vector"
IMPROVED_READER_MODE = "with_embedding_vector"
ONLY_EMBEDDING_READER_MODE = "only_embedding_vector"   # <--- update if your reader uses a different string

# Experiment IDs (FeatureReader internal experiment selector)
BASELINE_EXP_ID = 0
COMBINED_EXP_ID = 1
ONLY_EMBEDDING_EXP_ID = 2

DATA_ROOT_NAME = "Data_merge_embedding_features"

SIGNIFICANCE_LEVEL = 0.05
SHAP_POOLED_SAMPLE_SIZE = 5000

# Cache files
BASELINE_PKL_NAME = "baseline_predictions.pkl"
IMPROVED_PKL_NAME = "improved_predictions.pkl"
ONLY_EMBEDDING_PKL_NAME = "only_embedding_predictions.pkl"

# Outputs
DELONG_RESULTS_CSV = "delong_results.csv"
POOLED_SUMMARY_CSV = "pooled_delong_summary.csv"
THREE_MODELS_EPOCH_AUC_TABLE_CSV = "three_models_epoch_auc_table.csv"
THREE_MODELS_BEST_EPOCH_SUMMARY_CSV = "three_models_best_epoch_summary.csv"


# ============================================
#            DELONG IMPLEMENTATION
# ============================================

def roc_variance(ground_truth, predictions):
    """Compute AUC and Var(AUC) using DeLong-style U-statistic variance approximation."""
    ground_truth = np.array(ground_truth)
    predictions = np.array(predictions)

    pos = predictions[ground_truth == 1]
    neg = predictions[ground_truth == 0]

    m = len(pos)
    n = len(neg)
    if m == 0 or n == 0:
        raise ValueError("DeLong requires both positive and negative samples.")

    U = 0.0
    for p in pos:
        U += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    auc_val = U / (m * n)

    V10 = np.array([
        np.sum(pos[i] > neg) + 0.5 * np.sum(pos[i] == neg)
        for i in range(m)
    ]) / n

    V01 = np.array([
        np.sum(pos > neg[j]) + 0.5 * np.sum(pos == neg[j])
        for j in range(n)
    ]) / m

    var_auc = np.var(V10) / m + np.var(V01) / n
    return auc_val, var_auc


def delong_roc_test(y_true, pred1, pred2):
    """Returns: (auc1, auc2, p_value) comparing correlated ROC AUCs (same y_true)."""
    auc1, var1 = roc_variance(y_true, pred1)
    auc2, var2 = roc_variance(y_true, pred2)

    se = np.sqrt(var1 + var2)
    if se == 0:
        return auc1, auc2, 1.0

    z = (auc1 - auc2) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return auc1, auc2, p_value


# ============================================
#        PREDICTIONS COLLECTION PER MODEL
# ============================================

def _init_reader(reader_mode: str):
    FeatureReader.reader_selection_parameter = reader_mode
    fr = get_reader()
    if reader_mode == BASELINE_READER_MODE:
        fr.setnumbrtexperiments(exp_id=BASELINE_EXP_ID)
    elif reader_mode == IMPROVED_READER_MODE:
        fr.setnumbrtexperiments(exp_id=COMBINED_EXP_ID)
    elif reader_mode == ONLY_EMBEDDING_READER_MODE:
        fr.setnumbrtexperiments(exp_id=ONLY_EMBEDDING_EXP_ID)
    else:
        raise ValueError(f"Unknown reader_mode: {reader_mode}")
    return fr


def collect_predictions_for_model(
    base_dir: Path,
    model_subdir: str,
    reader_mode: str
):
    """
    Collect y_true and predicted probabilities per (fold, epoch) for a given model folder.
    """
    feature_reader = _init_reader(reader_mode)

    predictions = []
    model_root = base_dir / model_subdir
    data_root = base_dir / DATA_ROOT_NAME

    if not model_root.exists():
        print(f"[WARN] Missing model directory: {model_root}")
        return predictions

    for fold_path in sorted(model_root.glob("fold*")):
        fold_str = fold_path.name.replace("fold", "")
        try:
            fold = int(fold_str)
        except ValueError:
            print(f"[WARN] Invalid fold directory name: {fold_path}")
            continue

        for model_file in sorted(fold_path.glob("*.model")):
            parts = model_file.stem.split("_")
            if "epoch" in parts:
                idx = parts.index("epoch")
                epoch_str = parts[idx + 1]
            else:
                epoch_str = parts[3] if len(parts) > 3 else None

            if epoch_str is None:
                print(f"[WARN] Cannot parse epoch from {model_file.name}")
                continue

            try:
                epoch = int(epoch_str)
            except ValueError:
                print(f"[WARN] Cannot parse epoch from {model_file.name}")
                continue

            print(f"[RUN] {base_dir.name} — Fold {fold} Epoch {epoch} — {model_subdir}")

            with open(model_file, "rb") as f:
                model = pickle.load(f)

            test_csv = (
                data_root /
                f"fold{fold}" /
                f"epoch_{epoch}" /
                f"merged_test_fold_{fold}_epoch_{epoch}.csv"
            )
            if not test_csv.exists():
                print(f"[WARN] Missing test CSV: {test_csv}")
                continue

            X_test, y_test = feature_reader.file_reader(test_csv)
            y_prob = model.predict_proba(X_test)[:, 1]

            predictions.append({
                "fold": fold,
                "epoch": epoch,
                "y_true": y_test,
                "y_prob": y_prob
            })

            print(f"[OK] AUC={roc_auc_score(y_test, y_prob):.4f}")

    return predictions


def ensure_predictions_pkl(config_dir: Path):
    """
    Loads cached baseline/improved predictions if present.
    Otherwise computes and saves them.
    """
    base_pkl = config_dir / BASELINE_PKL_NAME
    imp_pkl = config_dir / IMPROVED_PKL_NAME

    if base_pkl.exists() and imp_pkl.exists():
        print(f"[INFO] Using existing prediction PKLs in {config_dir.name}")
        df_base = pd.read_pickle(base_pkl)
        df_improved = pd.read_pickle(imp_pkl)
        return df_base, df_improved

    print(f"[INFO] Collecting predictions for {config_dir.name}")

    baseline_preds = collect_predictions_for_model(
        config_dir,
        BASELINE_MODEL_DIR_NAME,
        BASELINE_READER_MODE
    )
    improved_preds = collect_predictions_for_model(
        config_dir,
        IMPROVED_MODEL_DIR_NAME,
        IMPROVED_READER_MODE
    )

    df_base = pd.DataFrame(baseline_preds)
    df_improved = pd.DataFrame(improved_preds)

    df_base.to_pickle(base_pkl)
    df_improved.to_pickle(imp_pkl)

    print(f"[INFO] Saved prediction PKLs in {config_dir.name}")
    return df_base, df_improved


def ensure_only_embedding_predictions_pkl(config_dir: Path):
    """
    Loads cached embedding-only predictions if present.
    Otherwise computes and saves them.
    """
    emb_pkl = config_dir / ONLY_EMBEDDING_PKL_NAME
    if emb_pkl.exists():
        print(f"[INFO] Using existing ONLY-EMBEDDING prediction PKL in {config_dir.name}")
        return pd.read_pickle(emb_pkl)

    print(f"[INFO] Collecting ONLY-EMBEDDING predictions for {config_dir.name}")

    emb_preds = collect_predictions_for_model(
        config_dir,
        ONLY_EMBEDDING_MODEL_DIR_NAME,
        ONLY_EMBEDDING_READER_MODE
    )
    df_emb = pd.DataFrame(emb_preds)
    df_emb.to_pickle(emb_pkl)
    print(f"[INFO] Saved ONLY-EMBEDDING prediction PKL in {config_dir.name}")
    return df_emb


# ============================================
#        DELONG + BEST EPOCH PER CONFIG
# ============================================

def compute_delong_per_fold_epoch(df_base: pd.DataFrame, df_improved: pd.DataFrame):
    """
    Merge baseline and combined by (fold, epoch) and compute:
    - auc_base, auc_improved
    - delta_auc
    - p_value per fold/epoch (DeLong)
    """
    merged = pd.merge(
        df_base,
        df_improved,
        on=["fold", "epoch"],
        suffixes=("_base", "_improved")
    )

    auc_base_list = []
    auc_improve_list = []
    pvals = []

    for _, row in merged.iterrows():
        y_true = row["y_true_base"]
        pb = row["y_prob_base"]
        pi = row["y_prob_improved"]

        auc_b = roc_auc_score(y_true, pb)
        auc_i = roc_auc_score(y_true, pi)
        auc_base_list.append(auc_b)
        auc_improve_list.append(auc_i)

        _, _, pval = delong_roc_test(y_true, pb, pi)
        pvals.append(pval)

    merged["auc_base"] = auc_base_list
    merged["auc_improved"] = auc_improve_list
    merged["delta_auc"] = merged["auc_improved"] - merged["auc_base"]
    merged["p_value"] = pvals

    return merged


def compute_epoch_level_stats(merged: pd.DataFrame):
    """
    Returns a per-epoch table containing:
    - mean_auc_base, mean_auc_improved, mean_delta (over folds)
    - pooled_auc_base, pooled_auc_improved, pooled_delta, pooled_p_value (DeLong on pooled)
    """
    epoch_stats = (
        merged
        .groupby("epoch")
        .agg(
            mean_auc_base=("auc_base", "mean"),
            mean_auc_improved=("auc_improved", "mean"),
            mean_delta=("delta_auc", "mean")
        )
        .reset_index()
    )

    pooled_rows = []
    for epoch, grp in merged.groupby("epoch"):
        y_true_pool = np.concatenate(grp["y_true_base"].values)
        pb_pool = np.concatenate(grp["y_prob_base"].values)
        pi_pool = np.concatenate(grp["y_prob_improved"].values)

        auc_b, auc_i, p_val = delong_roc_test(y_true_pool, pb_pool, pi_pool)
        pooled_rows.append({
            "epoch": int(epoch),
            "pooled_auc_base": float(auc_b),
            "pooled_auc_improved": float(auc_i),
            "pooled_delta": float(auc_i - auc_b),
            "pooled_p_value": float(p_val)
        })

    epoch_pooled = pd.DataFrame(pooled_rows)
    full_epoch_stats = pd.merge(epoch_stats, epoch_pooled, on="epoch")
    return full_epoch_stats


def select_best_epoch(full_epoch_stats: pd.DataFrame):
    """
    Select best epoch based on:
    - pooled_p_value < alpha
    - mean_delta > 0
    - maximize mean_delta
    """
    candidates = full_epoch_stats[
        (full_epoch_stats["pooled_p_value"] < SIGNIFICANCE_LEVEL) &
        (full_epoch_stats["mean_delta"] > 0)
    ]
    if candidates.empty:
        print("[INFO] No significant epoch found (p < 0.05 & ΔAUC>0).")
        return None

    best_idx = candidates["mean_delta"].idxmax()
    return full_epoch_stats.loc[best_idx]


# ============================================
#                PLOTS (ORIGINAL)
# ============================================

def plot_mean_delta_per_epoch(epoch_stats: pd.DataFrame, config_dir: Path):
    plt.figure(figsize=(8, 5))
    plt.plot(epoch_stats["epoch"], epoch_stats["mean_delta"], marker="o")
    plt.axhline(0, linestyle="--", color="black")
    plt.title("Mean ΔAUC per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Mean ΔAUC (combined - baseline)")
    plt.grid(True)
    out_path = config_dir / "mean_delta_auc_per_epoch.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved {out_path.name}")


def plot_pooled_roc(y_true_pool, pb_pool, pi_pool, auc_b, auc_i, best_epoch, config_dir: Path):
    fpr_b, tpr_b, _ = roc_curve(y_true_pool, pb_pool)
    fpr_i, tpr_i, _ = roc_curve(y_true_pool, pi_pool)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_b, tpr_b, label=f"Baseline AUC={auc_b:.3f}", lw=2, color="#1f77b4")
    plt.plot(fpr_i, tpr_i, label=f"GEMS AUC={auc_i:.3f}", lw=2, color="#d62728")

    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Pooled ROC Curve (Epoch {best_epoch})")
    plt.legend()
    plt.grid(True)
    out_path = config_dir / f"pooled_roc_epoch_{best_epoch}.png"
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved {out_path.name}")


def plot_delta_hist_for_epoch(merged: pd.DataFrame, best_epoch: int, config_dir: Path):
    rows = merged[merged["epoch"] == best_epoch]

    plt.figure(figsize=(8, 5))
    plt.hist(rows["delta_auc"], bins=8, edgecolor="black")
    plt.title(f"ΔAUC Distribution Across Folds (Epoch {best_epoch})")
    plt.xlabel("ΔAUC (combined - baseline)")
    plt.ylabel("Count")
    plt.grid(True)
    out_path = config_dir / f"delta_auc_hist_epoch_{best_epoch}.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved {out_path.name}")


def plot_baseline_vs_improved_epoch(epoch_stats: pd.DataFrame, config_dir: Path):
    plt.figure(figsize=(9, 6))
    plt.plot(epoch_stats["epoch"], epoch_stats["mean_auc_base"], marker="o", label="Baseline AUC")
    plt.plot(epoch_stats["epoch"], epoch_stats["mean_auc_improved"], marker="o", label="Combined AUC")
    plt.title("Baseline vs Combined AUC per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.grid(True)
    plt.legend()
    out_path = config_dir / "baseline_vs_improved_auc_per_epoch.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved {out_path.name}")


# ============================================
#      NEW: THREE-MODEL (BASE / EMB / COMBINED)
# ============================================

def compute_three_model_epoch_auc_table(df_base: pd.DataFrame,
                                        df_comb: pd.DataFrame,
                                        df_emb: pd.DataFrame) -> pd.DataFrame:
    """
    Per-epoch pooled AUC for Baseline / Embedding-only / Combined.
    Uses only epochs that exist in all three models.
    """
    epochs_base = set(df_base["epoch"].unique())
    epochs_comb = set(df_comb["epoch"].unique())
    epochs_emb  = set(df_emb["epoch"].unique())
    common_epochs = sorted(list(epochs_base & epochs_comb & epochs_emb))

    rows = []
    for epoch in common_epochs:
        b = df_base[df_base["epoch"] == epoch].sort_values("fold")
        c = df_comb[df_comb["epoch"] == epoch].sort_values("fold")
        e = df_emb[df_emb["epoch"] == epoch].sort_values("fold")

        bc = b.merge(c, on="fold", suffixes=("_base", "_combined"))
        bce = bc.merge(e, on="fold", suffixes=("", "_emb"))
        bce = bce.rename(columns={"y_prob": "y_prob_emb", "y_true": "y_true_emb"})

        y_pool = np.concatenate(bce["y_true_base"].values)
        pb = np.concatenate(bce["y_prob_base"].values)
        pc = np.concatenate(bce["y_prob_combined"].values)
        pe = np.concatenate(bce["y_prob_emb"].values)

        auc_b = roc_auc_score(y_pool, pb)
        auc_c = roc_auc_score(y_pool, pc)
        auc_e = roc_auc_score(y_pool, pe)

        rows.append({
            "epoch": int(epoch),
            "pooled_auc_baseline": float(auc_b),
            "pooled_auc_embedding_only": float(auc_e),
            "pooled_auc_combined": float(auc_c),
            "pooled_delta_combined_minus_baseline": float(auc_c - auc_b),
            "pooled_delta_emb_only_minus_baseline": float(auc_e - auc_b),
            "pooled_delta_combined_minus_emb_only": float(auc_c - auc_e),
        })

    return pd.DataFrame(rows).sort_values("epoch").reset_index(drop=True)


def plot_three_models_auc_per_epoch(three_epoch_auc: pd.DataFrame,
                                    config_dir: Path,
                                    best_epoch: int):
    plt.figure(figsize=(10, 6))
    plt.plot(three_epoch_auc["epoch"], three_epoch_auc["pooled_auc_baseline"], marker="o", label="Baseline")
    plt.plot(three_epoch_auc["epoch"], three_epoch_auc["pooled_auc_embedding_only"], marker="o", label="Embedding-only")
    plt.plot(three_epoch_auc["epoch"], three_epoch_auc["pooled_auc_combined"], marker="o", label="Combined")
    plt.axvline(best_epoch, linestyle="--", color="black", label=f"Best epoch = {best_epoch}")
    plt.title("Pooled AUC per Epoch — Three Models")
    plt.xlabel("Epoch")
    plt.ylabel("Pooled AUC")
    plt.grid(True)
    plt.legend()
    out_path = config_dir / "three_models_auc_per_epoch.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved {out_path.name}")


def plot_three_models_delta_auc_per_epoch(three_epoch_auc: pd.DataFrame,
                                          config_dir: Path,
                                          best_epoch: int):
    plt.figure(figsize=(10, 6))
    plt.plot(
        three_epoch_auc["epoch"],
        three_epoch_auc["pooled_delta_combined_minus_baseline"],
        marker="o",
        label="Combined - Baseline"
    )
    plt.plot(
        three_epoch_auc["epoch"],
        three_epoch_auc["pooled_delta_emb_only_minus_baseline"],
        marker="o",
        label="Embedding-only - Baseline"
    )
    plt.plot(
        three_epoch_auc["epoch"],
        three_epoch_auc["pooled_delta_combined_minus_emb_only"],
        marker="o",
        label="Combined - Embedding-only"
    )
    plt.axhline(0, linestyle="--", color="black")
    plt.axvline(best_epoch, linestyle="--", color="black", label=f"Best epoch = {best_epoch}")
    plt.title("Pooled ΔAUC per Epoch — Three Models")
    plt.xlabel("Epoch")
    plt.ylabel("Pooled ΔAUC")
    plt.grid(True)
    plt.legend()
    out_path = config_dir / "three_models_delta_auc_per_epoch.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved {out_path.name}")


def plot_three_models_pooled_roc_best_epoch(config_dir: Path, best_epoch: int,
                                            df_base: pd.DataFrame,
                                            df_comb: pd.DataFrame,
                                            df_emb: pd.DataFrame):
    """
    Pooled ROC curves for the 3 models at the selected best epoch.
    """
    b = df_base[df_base["epoch"] == best_epoch].sort_values("fold")
    c = df_comb[df_comb["epoch"] == best_epoch].sort_values("fold")
    e = df_emb[df_emb["epoch"] == best_epoch].sort_values("fold")

    bc = b.merge(c, on="fold", suffixes=("_base", "_combined"))
    bce = bc.merge(e, on="fold", suffixes=("", "_emb"))
    bce = bce.rename(columns={"y_prob": "y_prob_emb"})

    y_pool = np.concatenate(bce["y_true_base"].values)
    pb = np.concatenate(bce["y_prob_base"].values)
    pc = np.concatenate(bce["y_prob_combined"].values)
    pe = np.concatenate(bce["y_prob_emb"].values)

    auc_b = roc_auc_score(y_pool, pb)
    auc_c = roc_auc_score(y_pool, pc)
    auc_e = roc_auc_score(y_pool, pe)

    fpr_b, tpr_b, _ = roc_curve(y_pool, pb)
    fpr_c, tpr_c, _ = roc_curve(y_pool, pc)
    fpr_e, tpr_e, _ = roc_curve(y_pool, pe)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_b, tpr_b, label=f"Baseline AUC={auc_b:.3f}")
    plt.plot(fpr_e, tpr_e, label=f"Embedding-only AUC={auc_e:.3f}")
    plt.plot(fpr_c, tpr_c, label=f"Combined AUC={auc_c:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Pooled ROC — Three Models (Epoch {best_epoch})")
    plt.legend()
    plt.grid(True)
    out_path = config_dir / f"pooled_roc_three_models_epoch_{best_epoch}.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved {out_path.name}")


def plot_three_models_auc_distribution_best_epoch(config_dir: Path, best_epoch: int,
                                                  df_base: pd.DataFrame,
                                                  df_comb: pd.DataFrame,
                                                  df_emb: pd.DataFrame):
    """
    AUC distributions across folds at best epoch (box + violin).
    """
    # fold-level AUCs
    def _fold_auc(df):
        out = []
        for _, row in df.iterrows():
            out.append(roc_auc_score(row["y_true"], row["y_prob"]))
        return np.array(out, dtype=float)

    b = df_base[df_base["epoch"] == best_epoch].sort_values("fold")
    c = df_comb[df_comb["epoch"] == best_epoch].sort_values("fold")
    e = df_emb[df_emb["epoch"] == best_epoch].sort_values("fold")

    # ensure aligned folds (intersection)
    folds = sorted(list(set(b["fold"]) & set(c["fold"]) & set(e["fold"])))
    b = b[b["fold"].isin(folds)].sort_values("fold")
    c = c[c["fold"].isin(folds)].sort_values("fold")
    e = e[e["fold"].isin(folds)].sort_values("fold")

    auc_b = _fold_auc(b)
    auc_e = _fold_auc(e)
    auc_c = _fold_auc(c)

    # Boxplot
    plt.figure(figsize=(9, 6))
    plt.boxplot([auc_b, auc_e, auc_c], labels=["Baseline", "Embedding-only", "Combined"], showmeans=True)
    plt.title(f"AUC Distribution Across Folds (Epoch {best_epoch})")
    plt.ylabel("AUC")
    plt.grid(True, axis="y")
    out_path = config_dir / f"auc_boxplot_three_models_epoch_{best_epoch}.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved {out_path.name}")

    # Violin plot
    plt.figure(figsize=(9, 6))
    parts = plt.violinplot([auc_b, auc_e, auc_c], showmeans=True, showmedians=True)
    plt.xticks([1, 2, 3], ["Baseline", "Embedding-only", "Combined"])
    plt.title(f"AUC Violin Plot Across Folds (Epoch {best_epoch})")
    plt.ylabel("AUC")
    plt.grid(True, axis="y")
    out_path = config_dir / f"auc_violin_three_models_epoch_{best_epoch}.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved {out_path.name}")


def compute_three_models_delong_best_epoch(config_dir: Path, best_epoch: int,
                                           df_base: pd.DataFrame,
                                           df_comb: pd.DataFrame,
                                           df_emb: pd.DataFrame) -> dict:
    """
    Pooled DeLong comparisons at best epoch:
    - Combined vs Baseline
    - Embedding-only vs Baseline
    - Combined vs Embedding-only
    """
    b = df_base[df_base["epoch"] == best_epoch].sort_values("fold")
    c = df_comb[df_comb["epoch"] == best_epoch].sort_values("fold")
    e = df_emb[df_emb["epoch"] == best_epoch].sort_values("fold")

    folds = sorted(list(set(b["fold"]) & set(c["fold"]) & set(e["fold"])))
    b = b[b["fold"].isin(folds)].sort_values("fold")
    c = c[c["fold"].isin(folds)].sort_values("fold")
    e = e[e["fold"].isin(folds)].sort_values("fold")

    # pooled arrays
    y_pool = np.concatenate(b["y_true"].values)
    pb = np.concatenate(b["y_prob"].values)
    pc = np.concatenate(c["y_prob"].values)
    pe = np.concatenate(e["y_prob"].values)

    auc_b = roc_auc_score(y_pool, pb)
    auc_c = roc_auc_score(y_pool, pc)
    auc_e = roc_auc_score(y_pool, pe)

    # DeLong tests
    _, _, p_cb = delong_roc_test(y_pool, pc, pb)  # combined vs baseline
    _, _, p_eb = delong_roc_test(y_pool, pe, pb)  # emb vs baseline
    _, _, p_ce = delong_roc_test(y_pool, pc, pe)  # combined vs emb

    # correlations (optional reporting)
    rho_cb, _ = spearmanr(pc, pb)
    rho_ce, _ = spearmanr(pc, pe)
    rho_eb, _ = spearmanr(pe, pb)

    return {
        "config_dir": config_dir.name,
        "best_epoch": int(best_epoch),

        "auc_baseline": float(auc_b),
        "auc_embedding_only": float(auc_e),
        "auc_combined": float(auc_c),

        "delta_combined_minus_baseline": float(auc_c - auc_b),
        "delta_emb_only_minus_baseline": float(auc_e - auc_b),
        "delta_combined_minus_emb_only": float(auc_c - auc_e),

        "pvalue_combined_vs_baseline": float(p_cb),
        "pvalue_emb_only_vs_baseline": float(p_eb),
        "pvalue_combined_vs_emb_only": float(p_ce),

        "spearman_combined_vs_baseline": float(rho_cb),
        "spearman_combined_vs_emb_only": float(rho_ce),
        "spearman_emb_only_vs_baseline": float(rho_eb),

        "n_folds_used": int(len(folds)),
        "n_samples_pooled": int(len(y_pool)),
    }


# ============================================
#                SHAP (ORIGINAL, COMBINED ONLY)
# ============================================

def find_improved_model_file(config_dir: Path, fold: int, epoch: int) -> Path:
    model_dir = config_dir / IMPROVED_MODEL_DIR_NAME / f"fold{fold}"
    if not model_dir.exists():
        raise FileNotFoundError(f"Model dir not found: {model_dir}")

    pattern = f"fold_{fold}_epoch_{epoch}"
    candidates = [p for p in model_dir.glob("*.model") if pattern in p.name]
    if not candidates:
        raise FileNotFoundError(f"No model file matching {pattern} in {model_dir}")
    return candidates[0]


def load_test_data_for_fold_epoch(config_dir: Path, fold: int, epoch: int):
    FeatureReader.reader_selection_parameter = IMPROVED_READER_MODE
    feature_reader = get_reader()
    feature_reader.setnumbrtexperiments(exp_id=COMBINED_EXP_ID)

    test_csv = (
        config_dir / DATA_ROOT_NAME /
        f"fold{fold}" /
        f"epoch_{epoch}" /
        f"merged_test_fold_{fold}_epoch_{epoch}.csv"
    )
    if not test_csv.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")

    X_test, y_test = feature_reader.file_reader(test_csv)
    return X_test, y_test, test_csv


def shap_for_single_fold(config_dir: Path, best_epoch: int, example_fold: int):
    model_file = find_improved_model_file(config_dir, example_fold, best_epoch)
    with open(model_file, "rb") as f:
        shap_model = pickle.load(f)

    X_test, y_test, test_csv = load_test_data_for_fold_epoch(config_dir, example_fold, best_epoch)

    print(f"[SHAP] Single-fold — Config={config_dir.name}, Fold={example_fold}, Epoch={best_epoch}")
    print(f"[SHAP] Using model: {model_file.name}")
    print(f"[SHAP] Test CSV: {test_csv.name}, X_test shape={X_test.shape}")

    explainer = shap.TreeExplainer(shap_model)
    shap_values = explainer.shap_values(X_test)

    # If list (e.g., [class0, class1]) take class 1
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Summary plot
    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=X_test.columns,
        show=False
    )
    plt.title(f"SHAP Summary – Fold {example_fold}, Epoch {best_epoch}")
    plt.tight_layout()
    out1 = config_dir / f"shap_fold_summary_epoch_{best_epoch}_fold_{example_fold}.png"
    plt.savefig(out1, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] Saved {out1.name}")

    # Bar plot
    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=X_test.columns,
        plot_type="bar",
        show=False
    )
    out2 = config_dir / f"shap_fold_bar_epoch_{best_epoch}_fold_{example_fold}.png"
    plt.savefig(out2, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] Saved {out2.name}")


def shap_pooled_across_folds(config_dir: Path, best_epoch: int, folds: np.ndarray):
    all_X = []
    all_shap = []

    print(f"[SHAP-POOLED] Config={config_dir.name}, Epoch={best_epoch}")

    for fold in sorted(folds):
        try:
            model_file = find_improved_model_file(config_dir, fold, best_epoch)
        except FileNotFoundError as e:
            print(f"[SHAP-POOLED] Skipping fold {fold}: {e}")
            continue

        with open(model_file, "rb") as f:
            shap_model = pickle.load(f)

        X_test, y_test, test_csv = load_test_data_for_fold_epoch(config_dir, fold, best_epoch)
        print(f"[SHAP-POOLED] Fold {fold}: model={model_file.name}, test={test_csv.name}, n={len(X_test)}")

        explainer = shap.TreeExplainer(shap_model)
        shap_values = explainer.shap_values(X_test)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        all_X.append(X_test)
        all_shap.append(shap_values)

    if not all_X:
        print("[SHAP-POOLED] No folds collected for pooled SHAP.")
        return

    X_all = pd.concat(all_X, axis=0)
    shap_all = np.vstack(all_shap)

    n = X_all.shape[0]
    if n > SHAP_POOLED_SAMPLE_SIZE:
        idx = np.random.choice(n, SHAP_POOLED_SAMPLE_SIZE, replace=False)
        X_sample = X_all.iloc[idx]
        shap_sample = shap_all[idx]
    else:
        X_sample = X_all
        shap_sample = shap_all

    print(f"[SHAP-POOLED] Final sample size: {X_sample.shape[0]}")

    shap.summary_plot(
        shap_sample,
        X_sample,
        feature_names=X_sample.columns,
        show=False
    )
    plt.title(f"Pooled SHAP Summary – Epoch {best_epoch}")
    plt.tight_layout()
    out1 = config_dir / f"shap_pooled_summary_epoch_{best_epoch}.png"
    plt.savefig(out1, bbox_inches="tight")
    plt.close()
    print(f"[SHAP-POOLED] Saved {out1.name}")

    shap.summary_plot(
        shap_sample,
        X_sample,
        feature_names=X_sample.columns,
        plot_type="bar",
        show=False
    )
    out2 = config_dir / f"shap_pooled_bar_epoch_{best_epoch}.png"
    plt.savefig(out2, bbox_inches="tight")
    plt.close()
    print(f"[SHAP-POOLED] Saved {out2.name}")

def plot_mean_roc_two_models(df_base, df_comb, best_epoch, config_dir):

    fpr_grid = np.linspace(0,1,200)

    tprs_base = []
    tprs_comb = []

    aucs_base = []
    aucs_comb = []

    for fold in sorted(set(df_base["fold"])):
        b = df_base[(df_base.fold==fold)&(df_base.epoch==best_epoch)]
        c = df_comb[(df_comb.fold==fold)&(df_comb.epoch==best_epoch)]

        if b.empty or c.empty:
            continue

        y = b.iloc[0]["y_true"]

        fpr_b,tpr_b,_ = roc_curve(y, b.iloc[0]["y_prob"])
        fpr_c,tpr_c,_ = roc_curve(y, c.iloc[0]["y_prob"])

        tprs_base.append(np.interp(fpr_grid,fpr_b,tpr_b))
        tprs_comb.append(np.interp(fpr_grid,fpr_c,tpr_c))

        aucs_base.append(roc_auc_score(y,b.iloc[0]["y_prob"]))
        aucs_comb.append(roc_auc_score(y,c.iloc[0]["y_prob"]))

    mean_base = np.mean(tprs_base,axis=0)
    std_base = np.std(tprs_base,axis=0)

    mean_comb = np.mean(tprs_comb,axis=0)
    std_comb = np.std(tprs_comb,axis=0)

    auc_b = np.mean(aucs_base)
    auc_c = np.mean(aucs_comb)

    # pooled pvalue
    y_pool = np.concatenate(df_base[df_base.epoch==best_epoch]["y_true"])
    pb = np.concatenate(df_base[df_base.epoch==best_epoch]["y_prob"])
    pc = np.concatenate(df_comb[df_comb.epoch==best_epoch]["y_prob"])

    _,_,pval = delong_roc_test(y_pool,pc,pb)

    plt.figure(figsize=(8,6))

    plt.plot(fpr_grid,mean_base,color="#1f77b4",
             label=f"Baseline (AUC={auc_b:.3f})",lw=2)

    plt.fill_between(fpr_grid,
                     mean_base-std_base,
                     mean_base+std_base,
                     alpha=0.2,color="#1f77b4")

    plt.plot(fpr_grid,mean_comb,color="#d62728",
             label=f"Combined (AUC={auc_c:.3f})",lw=2)

    plt.fill_between(fpr_grid,
                     mean_comb-std_comb,
                     mean_comb+std_comb,
                     alpha=0.2,color="#d62728")

    plt.plot([0,1],[0,1],'--',color='black',lw=1)

    plt.title(f"Mean ROC across folds\nDeLong p-value={pval:.2e}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()

    out=config_dir/f"meanROC_two_models_epoch_{best_epoch}.png"
    plt.savefig(out)
    plt.close()
    print("[PLOT]",out.name)
def plot_mean_roc_three_models(df_base, df_comb, df_emb, best_epoch, config_dir):

    fpr_grid=np.linspace(0,1,200)

    store={"Baseline":[],
           "Embedding":[],
           "Combined":[]}

    aucs={"Baseline":[],
          "Embedding":[],
          "Combined":[]}

    for fold in sorted(set(df_base["fold"])):
        b=df_base[(df_base.fold==fold)&(df_base.epoch==best_epoch)]
        c=df_comb[(df_comb.fold==fold)&(df_comb.epoch==best_epoch)]
        e=df_emb[(df_emb.fold==fold)&(df_emb.epoch==best_epoch)]

        if b.empty or c.empty or e.empty:
            continue

        y=b.iloc[0]["y_true"]

        curves={
            "Baseline":roc_curve(y,b.iloc[0]["y_prob"]),
            "Combined":roc_curve(y,c.iloc[0]["y_prob"]),
            "Embedding":roc_curve(y,e.iloc[0]["y_prob"])
        }

        for k,(f,t,_) in curves.items():
            store[k].append(np.interp(fpr_grid,f,t))
            aucs[k].append(roc_auc_score(y,
                                         {"Baseline":b,
                                          "Combined":c,
                                          "Embedding":e}[k].iloc[0]["y_prob"]))

    colors={"Baseline":"#1f77b4",
            "Embedding":"#2ca02c",
            "Combined":"#d62728"}

    plt.figure(figsize=(8,6))

    for k in store:
        m=np.mean(store[k],axis=0)
        s=np.std(store[k],axis=0)

        plt.plot(fpr_grid,m,lw=2,
                 color=colors[k],
                 label=f"{k} (AUC={np.mean(aucs[k]):.3f})")

        plt.fill_between(fpr_grid,m-s,m+s,
                         alpha=0.15,color=colors[k])

    plt.plot([0,1],[0,1],'--',color='black')

    plt.title("Mean ROC across folds — Ablation comparison")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.tight_layout()

    out=config_dir/f"meanROC_three_models_epoch_{best_epoch}.png"
    plt.savefig(out)
    plt.close()
    print("[PLOT]",out.name)

# ============================================
#      PROCESS ONE CONFIGURATION END-TO-END
# ============================================

def process_single_configuration(config_dir: Path):
    print("\n" + "=" * 70)
    print(f"[CONFIG] Processing configuration: {config_dir.name}")
    print("=" * 70)

    # 1) Predictions: baseline + combined (original logic)
    df_base, df_improved = ensure_predictions_pkl(config_dir)

    # 1b) Predictions: embedding-only (new)
    df_emb = ensure_only_embedding_predictions_pkl(config_dir)

    if df_base.empty or df_improved.empty:
        print(f"[WARN] Empty baseline/combined predictions for {config_dir.name}, skipping.")
        return {
            "config_dir": config_dir.name,
            "status": "EMPTY",
            "best_epoch": None,
            "delta_auc": None,
            "p_value": None,
            "auc_base": None,
            "auc_improved": None,
        }

    # 2) DeLong baseline vs combined per fold/epoch (original)
    merged = compute_delong_per_fold_epoch(df_base, df_improved)

    # Save delong_results.csv
    delong_path = config_dir / DELONG_RESULTS_CSV
    merged.to_csv(delong_path, index=False)
    print(f"[INFO] Saved {delong_path.name}")

    # 3) Epoch-level stats + pooled per epoch (baseline vs combined)
    full_epoch_stats = compute_epoch_level_stats(merged)

    # Original plots
    plot_mean_delta_per_epoch(full_epoch_stats, config_dir)
    plot_baseline_vs_improved_epoch(full_epoch_stats, config_dir)

    # 4) Select best epoch (based ONLY on baseline vs combined, as you wanted)
    best_epoch_row = select_best_epoch(full_epoch_stats)
    if best_epoch_row is None:
        summary = {
            "config_dir": config_dir.name,
            "status": "NO_SIGNIFICANT_EPOCH",
            "best_epoch": None,
            "delta_auc": None,
            "p_value": None,
            "auc_base": None,
            "auc_improved": None,
        }
        pd.DataFrame([summary]).to_csv(config_dir / POOLED_SUMMARY_CSV, index=False)
        print(f"[INFO] No significant epoch in {config_dir.name}.")
        return summary

    best_epoch = int(best_epoch_row["epoch"])
    print(f"[BEST] Config={config_dir.name}, Best epoch={best_epoch}")
    print(best_epoch_row)

    # 5) Pooled DeLong for best epoch (baseline vs combined) (original)
    dfb_best = df_base[df_base["epoch"] == best_epoch]
    dfi_best = df_improved[df_improved["epoch"] == best_epoch]

    pooled = pd.merge(
        dfb_best,
        dfi_best,
        on="fold",
        suffixes=("_base", "_improved")
    )

    y_true_pool = np.concatenate(pooled["y_true_base"].values)
    pb_pool = np.concatenate(pooled["y_prob_base"].values)
    pi_pool = np.concatenate(pooled["y_prob_improved"].values)

    rho, _ = spearmanr(pb_pool, pi_pool)
    print(f"[INFO] Spearman correlation (pooled, best epoch): {rho:.4f}")

    auc_b, auc_i, p_val = delong_roc_test(y_true_pool, pb_pool, pi_pool)

    delta = auc_i - auc_b
    print("===== FINAL POOLED DELONG RESULT (best epoch) =====")
    print(f"Baseline AUC:    {auc_b:.4f}")
    print(f"Combined AUC:    {auc_i:.4f}")
    print(f"ΔAUC:            {delta:.4f}")
    print(f"p-value:         {p_val:.6f}")

    # Original pooled ROC + delta histogram
    plot_pooled_roc(y_true_pool, pb_pool, pi_pool, auc_b, auc_i, best_epoch, config_dir)
    plot_delta_hist_for_epoch(merged, best_epoch, config_dir)

    # 6) Save summary (original)
    summary = {
        "config_dir": config_dir.name,
        "status": "SIGNIFICANT" if p_val < SIGNIFICANCE_LEVEL else "NOT_SIGNIFICANT",
        "best_epoch": best_epoch,
        "delta_auc": float(delta),
        "p_value": float(p_val),
        "auc_base": float(auc_b),
        "auc_improved": float(auc_i),
    }
    pd.DataFrame([summary]).to_csv(config_dir / POOLED_SUMMARY_CSV, index=False)
    print(f"[INFO] Saved {POOLED_SUMMARY_CSV} in {config_dir.name}")

    # 7) SHAP (original, combined only)
    folds_available = merged["fold"].unique()
    example_fold = int(np.min(folds_available))

    try:
        shap_for_single_fold(config_dir, best_epoch, example_fold)
    except Exception as e:
        print(f"[SHAP] Error in single-fold SHAP: {e}")

    try:
        shap_pooled_across_folds(config_dir, best_epoch, folds_available)
    except Exception as e:
        print(f"[SHAP-POOLED] Error in pooled SHAP: {e}")

    # 8) NEW: three-model epoch curves (AUC and deltas) + best-epoch three-model plots
    if df_emb is None or df_emb.empty:
        print(f"[WARN] Embedding-only predictions empty in {config_dir.name}; skipping three-model plots.")
    else:
        three_epoch_auc = compute_three_model_epoch_auc_table(df_base, df_improved, df_emb)
        if three_epoch_auc.empty:
            print(f"[WARN] No common epochs across all 3 models in {config_dir.name}; skipping three-model epoch curves.")
        else:
            three_epoch_auc.to_csv(config_dir / THREE_MODELS_EPOCH_AUC_TABLE_CSV, index=False)
            print(f"[INFO] Saved {THREE_MODELS_EPOCH_AUC_TABLE_CSV} in {config_dir.name}")
            plot_three_models_auc_per_epoch(three_epoch_auc, config_dir, best_epoch)
            plot_three_models_delta_auc_per_epoch(three_epoch_auc, config_dir, best_epoch)

            plot_mean_roc_two_models(df_base, df_improved, best_epoch, config_dir)
            plot_mean_roc_three_models(df_base, df_improved, df_emb, best_epoch, config_dir)

        # Best epoch: three-model ROC + fold AUC distributions + pooled delong comparisons
        try:
            plot_three_models_pooled_roc_best_epoch(config_dir, best_epoch, df_base, df_improved, df_emb)
        except Exception as e:
            print(f"[WARN] Could not plot 3-model pooled ROC at best epoch: {e}")

        try:
            plot_three_models_auc_distribution_best_epoch(config_dir, best_epoch, df_base, df_improved, df_emb)
        except Exception as e:
            print(f"[WARN] Could not plot 3-model AUC distributions at best epoch: {e}")

        try:
            three_best = compute_three_models_delong_best_epoch(config_dir, best_epoch, df_base, df_improved, df_emb)
            pd.DataFrame([three_best]).to_csv(config_dir / THREE_MODELS_BEST_EPOCH_SUMMARY_CSV, index=False)
            print(f"[INFO] Saved {THREE_MODELS_BEST_EPOCH_SUMMARY_CSV} in {config_dir.name}")
        except Exception as e:
            print(f"[WARN] Could not compute 3-model DeLong summary at best epoch: {e}")

    print(f"[CONFIG DONE] {config_dir.name}")
    return summary


# ============================================
#         GLOBAL: ALL CONFIGURATIONS
# ============================================

def scan_all_configurations(root: Path):
    summaries = []
    for cfg_dir in sorted(root.glob("run_epochs*")):
        if not cfg_dir.is_dir():
            continue
        try:
            summary = process_single_configuration(cfg_dir)
            summaries.append(summary)
        except Exception as e:
            print(f"[ERROR] Failed processing {cfg_dir.name}: {e}")
            summaries.append({
                "config_dir": cfg_dir.name,
                "status": f"ERROR: {e}",
                "best_epoch": None,
                "delta_auc": None,
                "p_value": None,
                "auc_base": None,
                "auc_improved": None,
            })
    return pd.DataFrame(summaries)


def select_best_configuration_overall(all_df: pd.DataFrame):
    sig_df = all_df[all_df["status"] == "SIGNIFICANT"].dropna(subset=["delta_auc"])
    if not sig_df.empty:
        print("[GLOBAL] Selecting best configuration among SIGNIFICANT ones.")
        best_idx = sig_df["delta_auc"].idxmax()
        best = sig_df.loc[best_idx]
    else:
        print("[GLOBAL] No SIGNIFICANT configuration, falling back to max delta overall.")
        valid = all_df.dropna(subset=["delta_auc"])
        if valid.empty:
            print("[GLOBAL] No valid configurations at all.")
            return None
        best_idx = valid["delta_auc"].idxmax()
        best = valid.loc[best_idx]
    return best


def find_best_conf():
    print(f"[ROOT] Scanning experiments under: {EXPERIMENTS_ROOT}")
    all_df = scan_all_configurations(EXPERIMENTS_ROOT)

    out_all = EXPERIMENTS_ROOT / "all_configurations_summary.csv"
    all_df.to_csv(out_all, index=False)
    print(f"[GLOBAL] Saved all configurations summary to {out_all}")

    best_cfg = select_best_configuration_overall(all_df)
    if best_cfg is None:
        print("[GLOBAL] No best configuration could be determined.")
        return

    print("\n===== BEST CONFIGURATION OVERALL =====")
    print(f"Config directory: {best_cfg['config_dir']}")
    print(f"Status:           {best_cfg['status']}")
    print(f"Best epoch:       {best_cfg['best_epoch']}")
    print(f"ΔAUC (best):      {best_cfg['delta_auc']:.4f}")
    print(f"p-value:          {best_cfg['p_value']:.3e}")
    print(f"AUC base:         {best_cfg['auc_base']:.4f}")
    print(f"AUC combined:     {best_cfg['auc_improved']:.4f}")

    out_best = EXPERIMENTS_ROOT / "best_configuration_overall.csv"
    best_cfg.to_frame().T.to_csv(out_best, index=False)
    print(f"[GLOBAL] Saved best configuration to {out_best}")


# If you want to run:
# find_best_conf()


