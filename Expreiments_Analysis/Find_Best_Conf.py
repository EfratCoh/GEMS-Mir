import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve
from scipy import stats
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import shap

import Classifier.FeatureReader as FeatureReader
from Classifier.FeatureReader import get_reader


# ============================================
#            GLOBAL CONFIG
# ============================================

EXPERIMENTS_ROOT = Path(
    "/mnt/new_groups/vaksler_group/Efrat/Results/experiment_runs_featuers_new_try"
)

BASELINE_MODEL_DIR_NAME = "models_xgbs_without_embedding"
IMPROVED_MODEL_DIR_NAME = "models_xgbs_combine_embedding"

BASELINE_READER_MODE = "without_embedding_vector"
IMPROVED_READER_MODE = "with_embedding_vector"

DATA_ROOT_NAME = "Data_merge_embedding_features"

SIGNIFICANCE_LEVEL = 0.05
SHAP_POOLED_SAMPLE_SIZE = 5000

# קבצי שמות – נשמרים כמו שיש לכם כבר
BASELINE_PKL_NAME = "baseline_predictions.pkl"
IMPROVED_PKL_NAME = "improved_predictions.pkl"
DELONG_RESULTS_CSV = "delong_results.csv"
POOLED_SUMMARY_CSV = "pooled_delong_summary.csv"


# ============================================
#            DELONG IMPLEMENTATION
# ============================================

def roc_variance(ground_truth, predictions):
    """חישוב AUC ו-Var(AUC) לפי נוסחת DeLong."""
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
    """מחזיר: (auc1, auc2, p_value)"""
    auc1, var1 = roc_variance(y_true, pred1)
    auc2, var2 = roc_variance(y_true, pred2)

    se = np.sqrt(var1 + var2)
    if se == 0:
        # מקרה קיצון – אין שונות, אין מבחן
        return auc1, auc2, 1.0

    z = (auc1 - auc2) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return auc1, auc2, p_value


# ============================================
#        PREDICTIONS COLLECTION PER CONFIG
# ============================================

def collect_predictions_for_model(
    base_dir: Path,
    model_subdir: str,
    reader_mode: str
):
    """
    אוסף y_true ו-probabilities לכל (fold, epoch) עבור מודל אחד (baseline/improved).
    """
    FeatureReader.reader_selection_parameter = reader_mode
    feature_reader = get_reader()

    # הסתמכנו על הקוד שלך: exp_id = 0 ללא אימבדינג, 1 עם אימבדינג
    if reader_mode == BASELINE_READER_MODE:
        feature_reader.setnumbrtexperiments(exp_id=0)
    elif reader_mode == IMPROVED_READER_MODE:
        feature_reader.setnumbrtexperiments(exp_id=1)

    predictions = []

    model_root = base_dir / model_subdir
    data_root = base_dir / DATA_ROOT_NAME

    for fold_path in sorted(model_root.glob("fold*")):
        fold_str = fold_path.name.replace("fold", "")
        try:
            fold = int(fold_str)
        except ValueError:
            print(f"[WARN] Invalid fold directory name: {fold_path}")
            continue

        for model_file in sorted(fold_path.glob("*.model")):
            # לפי השם fold_1_epoch_99_xgbs.model
            parts = model_file.stem.split("_")
            # נחפש את המילה epoch
            if "epoch" in parts:
                idx = parts.index("epoch")
                epoch_str = parts[idx + 1]
            else:
                # fallback ישן: parts[3]
                epoch_str = parts[3]

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
    אם יש baseline/improved_predictions.pkl – טוען אותם.
    אחרת – מחשב ושומר.
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


# ============================================
#        DELONG + BEST EPOCH PER CONFIG
# ============================================

def compute_delong_per_fold_epoch(df_base: pd.DataFrame, df_improved: pd.DataFrame):
    """
    מחבר את הבייסליין והאימפרובד לפי (fold, epoch) ומחשב:
    - auc_base, auc_improved
    - delta_auc
    - p_value per fold/epoch
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
    מחזיר:
    - epoch_stats: mean aucs & mean delta
    - epoch_pooled: לכל epoch – pooled AUCs + pooled p-value
    """
    # mean per epoch (over folds)
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

    # pooled per epoch (concat כל הפולדים לתוך מבחן דילונג אחד)
    pooled_rows = []
    for epoch, grp in merged.groupby("epoch"):
        y_true_pool = np.concatenate(grp["y_true_base"].values)
        pb_pool = np.concatenate(grp["y_prob_base"].values)
        pi_pool = np.concatenate(grp["y_prob_improved"].values)

        auc_b, auc_i, p_val = delong_roc_test(y_true_pool, pb_pool, pi_pool)
        pooled_rows.append({
            "epoch": epoch,
            "pooled_auc_base": auc_b,
            "pooled_auc_improved": auc_i,
            "pooled_delta": auc_i - auc_b,
            "pooled_p_value": p_val
        })

    epoch_pooled = pd.DataFrame(pooled_rows)

    # מאחדים לטבלה אחת לנוחיות
    full_epoch_stats = pd.merge(epoch_stats, epoch_pooled, on="epoch")

    return full_epoch_stats


def select_best_epoch(full_epoch_stats: pd.DataFrame):
    """
    בוחר את ה-epoch הטוב ביותר לפי:
    - pooled_p_value < SIGNIFICANCE_LEVEL
    - mean_delta > 0
    ואז argmax(mean_delta)
    אם אין כזה – מחזיר None
    """
    candidates = full_epoch_stats[
        (full_epoch_stats["pooled_p_value"] < SIGNIFICANCE_LEVEL) &
        (full_epoch_stats["mean_delta"] > 0)
    ]

    if candidates.empty:
        print("[INFO] No significant epoch found (p < 0.05 & ΔAUC>0).")
        return None

    best_idx = candidates["mean_delta"].idxmax()
    best_row = full_epoch_stats.loc[best_idx]
    return best_row


# ============================================
#                PLOTS
# ============================================

def plot_mean_delta_per_epoch(epoch_stats: pd.DataFrame, config_dir: Path):
    plt.figure(figsize=(8, 5))
    plt.plot(epoch_stats["epoch"], epoch_stats["mean_delta"], marker="o")
    plt.axhline(0, linestyle="--", color="black")
    plt.title("Mean ΔAUC per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Mean ΔAUC (improved - baseline)")
    plt.grid(True)
    out_path = config_dir / "mean_delta_auc_per_epoch.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved {out_path.name}")


def plot_pooled_roc(y_true_pool, pb_pool, pi_pool, auc_b, auc_i, best_epoch, config_dir: Path):
    fpr_b, tpr_b, _ = roc_curve(y_true_pool, pb_pool)
    fpr_i, tpr_i, _ = roc_curve(y_true_pool, pi_pool)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_b, tpr_b, label=f"Baseline AUC={auc_b:.3f}")
    plt.plot(fpr_i, tpr_i, label=f"Improved AUC={auc_i:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Pooled ROC Curve (Epoch {best_epoch})")
    plt.legend()
    plt.grid(True)
    out_path = config_dir / f"pooled_roc_epoch_{best_epoch}.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved {out_path.name}")


def plot_delta_hist_for_epoch(merged: pd.DataFrame, best_epoch: int, config_dir: Path):
    rows = merged[merged["epoch"] == best_epoch]

    plt.figure(figsize=(8, 5))
    plt.hist(rows["delta_auc"], bins=8, edgecolor="black")
    plt.title(f"ΔAUC Distribution Across Folds (Epoch {best_epoch})")
    plt.xlabel("ΔAUC (improved - baseline)")
    plt.ylabel("Count")
    plt.grid(True)
    out_path = config_dir / f"delta_auc_hist_epoch_{best_epoch}.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved {out_path.name}")


def plot_baseline_vs_improved_epoch(epoch_stats: pd.DataFrame, config_dir: Path):
    plt.figure(figsize=(9, 6))
    plt.plot(epoch_stats["epoch"], epoch_stats["mean_auc_base"], marker="o", label="Baseline AUC")
    plt.plot(epoch_stats["epoch"], epoch_stats["mean_auc_improved"], marker="o", label="Improved AUC")
    plt.title("Baseline vs Improved AUC per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.grid(True)
    plt.legend()
    out_path = config_dir / "baseline_vs_improved_auc_per_epoch.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Saved {out_path.name}")


# ============================================
#                SHAP
# ============================================

def find_improved_model_file(config_dir: Path, fold: int, epoch: int) -> Path:
    """
    מוצא קובץ מודל ששייך ל-fold ול-epoch בתיקיית improved.
    """
    model_dir = config_dir / IMPROVED_MODEL_DIR_NAME / f"fold{fold}"
    if not model_dir.exists():
        raise FileNotFoundError(f"Model dir not found: {model_dir}")

    pattern = f"fold_{fold}_epoch_{epoch}"
    candidates = [p for p in model_dir.glob("*.model") if pattern in p.name]

    if not candidates:
        raise FileNotFoundError(f"No model file matching {pattern} in {model_dir}")

    return candidates[0]


def load_test_data_for_fold_epoch(config_dir: Path, fold: int, epoch: int):
    """
    טוען X_test, y_test עבור fold & epoch.
    """
    FeatureReader.reader_selection_parameter = IMPROVED_READER_MODE
    feature_reader = get_reader()
    feature_reader.setnumbrtexperiments(exp_id=1)

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
    """
    SHAP על Fold אחד (improved model בלבד) עבור ה-best epoch.
    """
    model_file = find_improved_model_file(config_dir, example_fold, best_epoch)

    with open(model_file, "rb") as f:
        shap_model = pickle.load(f)

    X_test, y_test, test_csv = load_test_data_for_fold_epoch(config_dir, example_fold, best_epoch)

    print(f"[SHAP] Single-fold — Config={config_dir.name}, Fold={example_fold}, Epoch={best_epoch}")
    print(f"[SHAP] Using model: {model_file.name}")
    print(f"[SHAP] Test CSV: {test_csv.name}, X_test shape={X_test.shape}")

    explainer = shap.TreeExplainer(shap_model)
    shap_values = explainer.shap_values(X_test)

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
    """
    SHAP pooled: מאחדים X_test + SHAP מכל הפולדים עבור ה-best epoch.
    מדגמים עד SHAP_POOLED_SAMPLE_SIZE.
    """
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

        # אם המודל שלך מחזיר shap_values כ-list (למשל [class0, class1]), נניח שאת רוצה את של class 1:
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        all_X.append(X_test)
        all_shap.append(shap_values)

    if not all_X:
        print("[SHAP-POOLED] No folds collected for pooled SHAP.")
        return

    X_all = pd.concat(all_X, axis=0)
    shap_all = np.vstack(all_shap)

    # דגימה אם גדול מדי
    n = X_all.shape[0]
    if n > SHAP_POOLED_SAMPLE_SIZE:
        idx = np.random.choice(n, SHAP_POOLED_SAMPLE_SIZE, replace=False)
        X_sample = X_all.iloc[idx]
        shap_sample = shap_all[idx]
    else:
        X_sample = X_all
        shap_sample = shap_all

    print(f"[SHAP-POOLED] Final sample size: {X_sample.shape[0]}")

    # Summary plot
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

    # Bar plot
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


# ============================================
#      PROCESS ONE CONFIGURATION END-TO-END
# ============================================

def process_single_configuration(config_dir: Path):
    """
    מריץ את כל הניתוחים עבור קונפיגורציה אחת.
    מחזיר dict לסיכום ברמת ה-root.
    """
    print("\n" + "=" * 70)
    print(f"[CONFIG] Processing configuration: {config_dir.name}")
    print("=" * 70)

    # 1) Predictions
    df_base, df_improved = ensure_predictions_pkl(config_dir)

    if df_base.empty or df_improved.empty:
        print(f"[WARN] Empty predictions for {config_dir.name}, skipping.")
        return {
            "config_dir": config_dir.name,
            "status": "EMPTY",
            "best_epoch": None,
            "delta_auc": None,
            "p_value": None,
            "auc_base": None,
            "auc_improved": None,
        }

    # 2) Delong per fold/epoch
    merged = compute_delong_per_fold_epoch(df_base, df_improved)

    # Save delong_results.csv
    delong_path = config_dir / DELONG_RESULTS_CSV
    merged.to_csv(delong_path, index=False)
    print(f"[INFO] Saved {delong_path.name}")

    # 3) Epoch-level stats + pooled per epoch
    full_epoch_stats = compute_epoch_level_stats(merged)

    # Plots that לא תלויים בבחירת epoch
    plot_mean_delta_per_epoch(full_epoch_stats, config_dir)
    plot_baseline_vs_improved_epoch(full_epoch_stats, config_dir)

    # 4) Select best epoch
    best_epoch_row = select_best_epoch(full_epoch_stats)
    if best_epoch_row is None:
        # אין אפוק מובהק – שומרים סיכום קטן ונפלט החוצה
        summary = {
            "config_dir": config_dir.name,
            "status": "NO_SIGNIFICANT_EPOCH",
            "best_epoch": None,
            "delta_auc": None,
            "p_value": None,
            "auc_base": None,
            "auc_improved": None,
        }
        # עדיין אפשר לשמור summary קטן
        pd.DataFrame([summary]).to_csv(config_dir / POOLED_SUMMARY_CSV, index=False)
        print(f"[INFO] No significant epoch in {config_dir.name}.")
        return summary

    best_epoch = int(best_epoch_row["epoch"])
    print(f"[BEST] Config={config_dir.name}, Best epoch={best_epoch}")
    print(best_epoch_row)

    # 5) Pooled DeLong עבור ה-best epoch (בפועל יש לנו כבר בשורה)
    # אנחנו רוצים גם לראות את y_true, pb, pi pooled
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
    print(f"Improved AUC:    {auc_i:.4f}")
    print(f"ΔAUC:            {delta:.4f}")
    print(f"p-value:         {p_val:.6f}")

    # Plot pooled ROC & hist for best epoch
    plot_pooled_roc(y_true_pool, pb_pool, pi_pool, auc_b, auc_i, best_epoch, config_dir)
    plot_delta_hist_for_epoch(merged, best_epoch, config_dir)

    # 6) Save summary
    summary = {
        "config_dir": config_dir.name,
        "status": "SIGNIFICANT" if p_val < SIGNIFICANCE_LEVEL else "NOT_SIGNIFICANT",
        "best_epoch": best_epoch,
        "delta_auc": delta,
        "p_value": p_val,
        "auc_base": auc_b,
        "auc_improved": auc_i,
    }

    pd.DataFrame([summary]).to_csv(config_dir / POOLED_SUMMARY_CSV, index=False)
    print(f"[INFO] Saved {POOLED_SUMMARY_CSV} in {config_dir.name}")

    # 7) SHAP – only if יש best_epoch
    folds_available = merged["fold"].unique()
    example_fold = int(np.min(folds_available))

    # single-fold shap
    try:
        shap_for_single_fold(config_dir, best_epoch, example_fold)
    except Exception as e:
        print(f"[SHAP] Error in single-fold SHAP: {e}")

    # pooled shap
    try:
        shap_pooled_across_folds(config_dir, best_epoch, folds_available)
    except Exception as e:
        print(f"[SHAP-POOLED] Error in pooled SHAP: {e}")

    print(f"[CONFIG DONE] {config_dir.name}")
    return summary


# ============================================
#         GLOBAL: ALL CONFIGURATIONS
# ============================================

def scan_all_configurations(root: Path):
    """
    רץ על כל התיקיות שהשם שלהן מתחיל ב-run_epochs ומחזיר DataFrame של summaries.
    """
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

    all_df = pd.DataFrame(summaries)
    return all_df


def select_best_configuration_overall(all_df: pd.DataFrame):
    """
    בוחר את הקונפיגורציה הטובה ביותר לפי:
    - קודם כל רק אלה עם status == "SIGNIFICANT"
    - אם אין – לוקח max(delta_auc) מהכל
    """
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
    print(f"AUC improved:     {best_cfg['auc_improved']:.4f}")

    out_best = EXPERIMENTS_ROOT / "best_configuration_overall.csv"
    best_cfg.to_frame().T.to_csv(out_best, index=False)
    print(f"[GLOBAL] Saved best configuration to {out_best}")



