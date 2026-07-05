import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import Classifier.FeatureReader as FeatureReader
from Classifier.FeatureReader import get_reader
from consts.global_consts import ROOT_PATH_PHD_GOAL_ONE,DATA_TRAIN_TEST, NEGATIVE_DATA_PATH, MERGE_DATA, DATA_PATH_INTERACTIONS
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# =========================
#       DELONG TEST
# =========================

def roc_variance(ground_truth, predictions):
    ground_truth = np.array(ground_truth)
    predictions = np.array(predictions)

    pos = predictions[ground_truth == 1]
    neg = predictions[ground_truth == 0]

    m = len(pos)
    n = len(neg)

    U = 0
    for p in pos:
        U += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    auc = U / (m * n)

    V10 = np.array([np.sum(pos[i] > neg) +
                    0.5 * np.sum(pos[i] == neg) for i in range(m)]) / n
    V01 = np.array([np.sum(pos > neg[j]) +
                    0.5 * np.sum(pos == neg[j]) for j in range(n)]) / m

    var_auc = (np.var(V10) / m) + (np.var(V01) / n)
    return auc, var_auc


def delong_roc_test(y_true, y_pred1, y_pred2):
    auc1, var1 = roc_variance(y_true, y_pred1)
    auc2, var2 = roc_variance(y_true, y_pred2)

    se = np.sqrt(var1 + var2)
    z = (auc1 - auc2) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return p_value



def collect_predictions(base_dir, model_mode, reader_mode):
    FeatureReader.reader_selection_parameter = reader_mode
    feature_reader = get_reader()

    predictions = []

    model_root = Path(base_dir) / model_mode
    data_root =  Path(base_dir) / "Data_merge_embedding_features" #folder of the data test


    for fold_path in sorted(model_root.glob("fold*")):
        fold = fold_path.name.replace("fold", "")

        for model_file in sorted(fold_path.glob("*.model")):
            epoch = model_file.stem.split("_")[3]

            # load model
            with open(model_file, "rb") as f:
                model = pickle.load(f)

            # locate test csv
            test_csv = data_root / f"fold{fold}" / f"epoch_{epoch}" / f"merged_test_fold_{fold}_epoch_{epoch}.csv"
            if not test_csv.exists():
                print("Missing test:", test_csv)
                continue

            X_test, y_test = feature_reader.file_reader(test_csv)

            y_prob = model.predict_proba(X_test)[:, 1]

            predictions.append({
                "fold": int(fold),
                "epoch": int(epoch),
                "model_mode": model_mode,
                "y_true": y_test,
                "y_prob": y_prob
            })

            print(f"[OK] Fold {fold} Epoch {epoch} — {model_mode} — AUC={roc_auc_score(y_test, y_prob):.4f}")

    return predictions



base_dir = "/mnt/new_groups/vaksler_group/Efrat/Results/experiment_runs_16_featuers/run_epochs_80.0_lr_0.006_din_128.0_dout_256.0_bs_64.0/"


print("[INFO] Evaluating results...")
improved_predictions = collect_predictions(base_dir=base_dir,
                model_mode="models_xgbs_combine_embedding",
                reader_mode="with_embedding_vector")

baseline_predictions = collect_predictions(base_dir=base_dir,
                model_mode="models_xgbs_without_embedding",
                reader_mode="without_embedding_vector")

# Convert to DF and save
df_base = pd.DataFrame(baseline_predictions)
df_improved = pd.DataFrame(improved_predictions)
print(df_base)
print("Dddd")

df_base.to_pickle(Path(base_dir) / "baseline_predictions.pkl")
df_improved.to_pickle(Path(base_dir) / "improved_predictions.pkl")





print("Loading saved prediction files...")

df_base = pd.read_pickle(Path(base_dir) / "baseline_predictions.pkl")
df_improved = pd.read_pickle(Path(base_dir) / "improved_predictions.pkl")

print("Loaded baseline_predictions.pkl and improved_predictions.pkl")

# =========================
#       DELONG ANALYSIS
# =========================

merged = pd.merge(
    df_base,
    df_improved,
    on=["fold", "epoch"],
    suffixes=("_base", "_improved")
)

p_values = []

for _, row in merged.iterrows():
    y_true = row["y_true_base"]
    prob_base = row["y_prob_base"]
    prob_improved = row["y_prob_improved"]

    p = delong_roc_test(y_true, prob_base, prob_improved)
    p_values.append(p)

merged["p_value"] = p_values
print(p_values)
merged.to_csv(Path(base_dir) / "delong_results.csv", index=False)
print("Done! Saved: delong_results.csv")


import pandas as pd

# טוען את קובץ DeLong
df = pd.read_csv(Path(base_dir) / "delong_results.csv")

print("\n===== קובץ DeLong נטען בהצלחה =====\n")

# תנאי למובהקות
significance_threshold = 0.05

# מונה כמה folds עבור כל epoch הם מובהקים
epoch_significant_counts = (
    df.groupby("epoch")["p_value"]
      .apply(lambda x: (x < significance_threshold).sum())
)

print("===== מספר הפולדים עם p<0.05 בכל epoch =====")
for epoch, count in epoch_significant_counts.items():
    print(f"Epoch {epoch}: {count} פולדים מובהקים")

# מזהה epochs טובים: לפחות 9/10
good_epochs = epoch_significant_counts[epoch_significant_counts >= 9]

print("\n===== חיפוש epoch עם מובהקות רוחבית (>= 9/10) =====")

if len(good_epochs) == 0:
    print("❌ לא נמצא אף epoch שבו לפחות 9 מתוך 10 folds מובהקים סטטיסטית.")
else:
    print("✔ נמצאו epochs בעלי מובהקות גבוהה:")
    for epoch, count in good_epochs.items():
        print(f"➡ Epoch {epoch}: {count}/10 folds מובהקים (p < 0.05)")

    # בוחרים את הטוב ביותר: זה עם מספר המובהקים הגבוה ביותר
    best_epoch = good_epochs.idxmax()
    best_count = good_epochs.max()

    print("\n===== ה-Epoch הטוב ביותר =====")
    print(f"⭐ Epoch {best_epoch} הוא החזק ביותר — {best_count}/10 folds מובהקים!")
