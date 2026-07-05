import os
import pandas as pd
from pathlib import Path

# הנתיב לכל תיקיות הניסוי
base_path = "/sise/vaksler-group/IsanaRNA/Efrat/Results/experiment_runs/"

# רשימת תוצאות
best_epochs = []

for folder_name in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder_name)
    results_path = os.path.join(folder_path, "Results")

    if not os.path.isdir(results_path):
        continue

    try:
        # קריאה
        with_path = os.path.join(results_path, "evaluation_summary_models_xgbs_combine_embedding.csv")
        without_path = os.path.join(results_path, "evaluation_summary_models_xgbs_without_embedding.csv")

        df_with = pd.read_csv(with_path)
        df_without = pd.read_csv(without_path)

        # מיזוג לפי fold+epoch
        merged = pd.merge(df_with, df_without, on=["fold", "epoch"], suffixes=('_with', '_without'))

        # חישוב השיפור
        merged["auc_delta"] = merged["auc_with"] - merged["auc_without"]

        # חישוב ממוצע AUC עם embedding לכל epoch
        grouped = merged.groupby("epoch").agg({
            "auc_with": "mean",
            "auc_without": "mean",
            "auc_delta": ["mean", lambda x: (x > 0).sum()]
        }).reset_index()

        grouped.columns = ["epoch", "avg_auc_with", "avg_auc_without", "avg_delta_auc", "num_folds_improved"]

        # בחירת ה־epoch עם ממוצע AUC הגבוה ביותר עם embedding
        best_row = grouped.loc[grouped["avg_auc_with"].idxmax()]

        # שמירה
        best_epochs.append({
            "folder": folder_name,
            "best_epoch": int(best_row["epoch"]),
            "avg_auc_with": best_row["avg_auc_with"],
            "avg_auc_without": best_row["avg_auc_without"],
            "avg_auc_delta": best_row["avg_delta_auc"],
            "num_folds_improved": int(best_row["num_folds_improved"])
        })

    except Exception as e:
        print(f"שגיאה בקריאת {folder_name}: {e}")
        continue

# הפיכה לטבלה
df_summary = pd.DataFrame(best_epochs)



path_save = Path(base_path) / "embedding_auc_analysis.csv"
# שמירה לקובץ CSV
df_summary.to_csv(path_save, index=False)

# # הדפסת השורות הכי טובות לכל ניסוי
# best_epochs_df = df_summary[df_summary["is_best_epoch"]]
# print(best_epochs_df[["folder", "epoch", "improved_folds", "avg_auc_with", "avg_auc_without", "avg_auc_delta"]])
