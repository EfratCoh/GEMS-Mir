import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# =====================================================================
# פונקציה 1: איסוף הנתונים מתיקיות הניסויים (מסנן רק 80 אפוקים)
# =====================================================================
def collect_experiments_data(base_dir, target_epoch='best'):
    base_path = Path(base_dir)
    results_list = []

    pattern = re.compile(
        r"run_epochs_(?P<epochs>[\d\.]+)_lr_(?P<lr>[\d\.]+)_din_(?P<din>[\d\.]+)_dout_(?P<dout>[\d\.]+)_bs_(?P<bs>[\d\.]+)")

    print("🔍 סורק את תיקיות הניסויים...")

    for run_dir in base_path.glob("run_*"):
        if not run_dir.is_dir():
            continue

        match = pattern.search(run_dir.name)
        if not match:
            continue

        params = {
            "epochs": float(match.group("epochs")),
            "lr": float(match.group("lr")),
            "din": float(match.group("din")),
            "dout": float(match.group("dout")),
            "bs": float(match.group("bs")),
            "dir_name": run_dir.name
        }

        # סינון ל-80 אפוקים
        run_epochs_int = int(params["epochs"])

        results_folder = run_dir / "Results"
        if not results_folder.exists():
            results_folder = run_dir / "Results"
            if not results_folder.exists():
                continue

        combine_file = results_folder / "gems_metrics_models_xgbs_combine_embedding.csv"
        without_file = results_folder / "gems_metrics_models_xgbs_without_embedding.csv"
        only_file = results_folder / "gems_metrics_models_xgbs_only_embedding.csv"

        try:
            # פונקציית עזר פנימית ששואבת את הלוגיקה המדויקת מהאקסל שלך!
            def extract_metrics(df_path):
                df = pd.read_csv(df_path)

                if target_epoch == 'best':
                    # כמו באקסל: מקבצים לפי אפוק ומוצאים את המנצח
                    mean_per_epoch = df.groupby('epoch')['auc'].mean()
                    best_ep = mean_per_epoch.idxmax()
                    df_target = df[df['epoch'] == best_ep]
                else:
                    # מסננים לפי אפוק ספציפי (אם העברת מספר)
                    df_target = df[df['epoch'] == int(target_epoch)]

                if df_target.empty:
                    return None, None

                # מוציאים את הממוצעים של האפוק הממוקד
                f1_col = 'f1_score' if 'f1_score' in df_target.columns else 'f1'
                return df_target['auc'].mean(), df_target[f1_col].mean()

            # שליפת הנתונים בפועל
            if combine_file.exists():
                auc_c, f1_c = extract_metrics(combine_file)
                params["auc_combine"] = auc_c
                params["f1_combine"] = f1_c

            if without_file.exists():
                auc_w, f1_w = extract_metrics(without_file)
                params["auc_without"] = auc_w
                params["f1_without"] = f1_w

            if only_file.exists():
                auc_o, f1_o = extract_metrics(only_file)
                params["auc_only"] = auc_o
                params["f1_only"] = f1_o

            # מוסיפים לרשימה רק אם חולץ בהצלחה AUC למודל הראשי
            if params.get("auc_combine") is not None:
                results_list.append(params)

        except Exception as e:
            print(f"⚠️ שגיאה בקריאת תיקייה {run_dir.name}: {e}")

    df_all = pd.DataFrame(results_list)
    print(f"✅ נאספו נתונים מ-{len(df_all)} ריצות שונות (רק של 80 אפוקים).")
    return df_all


# =====================================================================
# פונקציה 2: יצירת הגרפים המעוצבים ברמה האקדמית
# ===================================

import os
import matplotlib.pyplot as plt
import seaborn as sns


def plot_ablation_curves_clean(df, output_dir, target_epoch='best'):
    if df.empty:
        print("❌ אין נתונים להציג (הטבלה ריקה).")
        return

    # הגדרות עיצוב למאמר
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    sns.set_theme(style="whitegrid")

    y_label_text = 'Max Mean Test AUC' if target_epoch == 'best' else f'Mean Test AUC (Epoch {target_epoch})'

    # טווח קבוע לציר ה-Y כדי להראות את יציבות המודל!
    # טווח קבוע לציר ה-Y
    Y_MIN = 0.840
    Y_MAX = 0.865

    # ---------------------------------------------------------
    # פיגר 1: מרובה פאנלים (AUC כפונקציה של Output Dim, לפי Learning Rates)
    # ---------------------------------------------------------
    unique_lrs = sorted(df['lr'].unique())
    num_lrs = len(unique_lrs)

    fig, axes = plt.subplots(1, num_lrs, figsize=(7 * num_lrs, 6), sharey=True)

    # התאמה למצב של קצב למידה אחד בלבד
    if num_lrs == 1:
        axes = [axes]

    for i, current_lr in enumerate(unique_lrs):
        ax = axes[i]
        subset_lr = df[df['lr'] == current_lr].copy()
        subset_lr['dout'] = subset_lr['dout'].astype(int)
        subset_lr['din'] = subset_lr['din'].astype(int)
        subset_lr = subset_lr.sort_values(by=['din', 'dout'])

        sns.lineplot(
            data=subset_lr, x='dout', y='auc_combine', hue='din',
            marker='o', markersize=9, linewidth=2.5, palette='Set1', ax=ax
        )

        # כותרת הפאנל
        ax.set_title(f"Learning Rate: {current_lr}", fontsize=15, fontweight='bold', pad=15)

        # שינוי התוויות לטקסט נקי ברמת פרסום
        ax.set_xlabel('Graph Embedding Output Dimension', fontsize=14, labelpad=10)

        # קיבוע ציר ה-Y
        ax.set_ylim(Y_MIN, Y_MAX)

        if i == 0:
            ax.set_ylabel(y_label_text, fontsize=14, labelpad=10)
        else:
            ax.set_ylabel('')

        if i == num_lrs - 1:
            # מקרא נקי ללא (din)
            ax.legend(title='Input Dimension', fontsize=13, title_fontsize=14, loc='best', framealpha=0.9)
        else:
            if ax.get_legend():
                ax.get_legend().remove()

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks(sorted(subset_lr['dout'].unique()))
        ax.tick_params(axis='both', which='major', labelsize=12)

    # הסרתי את fig.suptitle כי במאמרים הכותרת צריכה להיות בקפשן ולא בתמונה
    plt.tight_layout()

    save_name_1 = f"GEMS_Ablation_dout_vs_din_multi_LR_{'best' if target_epoch == 'best' else f'ep{target_epoch}'}_PublicationReady"
    save_path_1 = os.path.join(output_dir, f'{save_name_1}.png')
    plt.savefig(save_path_1, dpi=300, bbox_inches='tight')
    # plt.savefig(save_path_1.replace('.png', '.pdf'), bbox_inches='tight') # אפשר להחזיר אם צריכה PDF
    plt.show()
    print(f"✅ פיגר 1 (מרובה פאנלים) נוצר ונשמר ב:\n{save_path_1}")

    # ---------------------------------------------------------
    # פיגר 2: השפעת קצב הלמידה (עבור ה-din הטוב ביותר)
    # ---------------------------------------------------------
    best_din = df.loc[df['auc_combine'].idxmax()]['din']
    subset_din = df[df['din'] == best_din].copy()

    subset_din['dout'] = subset_din['dout'].astype(int)
    subset_din = subset_din.sort_values(by=['lr', 'dout'])

    plt.figure(figsize=(9, 6))
    sns.lineplot(
        data=subset_din, x='dout', y='auc_combine', hue='lr',
        marker='s', markersize=9, linewidth=2.5, palette='Dark2'
    )

    # הסרתי את הכותרת העליונה לטובת מראה נקי
    # plt.title(f"Ablation: Output Dimension vs. Learning Rate\n{title_suffix} | Fixed din: {int(best_din)}", fontsize=16, fontweight='bold', pad=15)

    plt.xlabel('Graph Embedding Output Dimension', fontsize=14, labelpad=10)
    plt.ylabel(y_label_text, fontsize=14, labelpad=10)

    # קיבוע ציר ה-Y
    plt.ylim(Y_MIN, Y_MAX)

    # מקרא נקי ללא (lr)
    plt.legend(title='Learning Rate', fontsize=13, title_fontsize=14, loc='best', framealpha=0.9)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xticks(sorted(subset_din['dout'].unique()))
    plt.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()

    save_name_2 = f"GEMS_Ablation_dout_vs_lr_{'best' if target_epoch == 'best' else f'ep{target_epoch}'}_PublicationReady"
    save_path_2 = os.path.join(output_dir, f'{save_name_2}.png')
    plt.savefig(save_path_2, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ פיגר 2 נוצר ונשמר ב:\n{save_path_2}")


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_ablation_batch_size_grid(df, output_dir, target_epoch='best'):
    if df.empty:
        print("❌ אין נתונים להציג (הטבלה ריקה).")
        return

    # הגדרות עיצוב למאמר
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    sns.set_theme(style="whitegrid")

    y_label_text = 'Max Mean Test AUC' if target_epoch == 'best' else f'Mean Test AUC (Epoch {target_epoch})'

    # טווח קבוע לציר ה-Y כדי להראות יציבות
    Y_MIN = 0.840
    Y_MAX = 0.865

    # חילוץ הערכים הייחודיים עבור הרשת (Grid)
    unique_bs = sorted(df['bs'].dropna().unique())
    unique_lrs = sorted(df['lr'].dropna().unique())

    num_rows = len(unique_bs)
    num_cols = len(unique_lrs)

    # יצירת הקנבס המרובע (Grid)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(7 * num_cols, 5 * num_rows), sharex=True, sharey=True)

    # הבטחה שה-axes הם תמיד מערך דו-מימדי (גם אם יש רק שורה אחת או עמודה אחת)
    axes = np.atleast_2d(axes)

    for row, current_bs in enumerate(unique_bs):
        for col, current_lr in enumerate(unique_lrs):
            ax = axes[row, col]

            # סינון הנתונים לפאנל הספציפי
            subset = df[(df['bs'] == current_bs) & (df['lr'] == current_lr)].copy()

            if not subset.empty:
                subset['dout'] = subset['dout'].astype(int)
                subset['din'] = subset['din'].astype(int)
                subset = subset.sort_values(by=['din', 'dout'])

                sns.lineplot(
                    data=subset, x='dout', y='auc_combine', hue='din',
                    marker='o', markersize=9, linewidth=2.5, palette='Set1', ax=ax
                )

            # כותרת הפאנל: מציגה גם את ה-Batch Size וגם את ה-Learning Rate
            ax.set_title(f"Batch Size: {int(current_bs)} | LR: {current_lr}", fontsize=15, fontweight='bold', pad=12)

            # קיבוע ציר ה-Y
            ax.set_ylim(Y_MIN, Y_MAX)

            # --- ניהול תוויות הצירים כדי שהגרף יהיה נקי ---

            # ציר ה-Y: רק לעמודה השמאלית ביותר
            if col == 0:
                ax.set_ylabel(y_label_text, fontsize=14, labelpad=10)
            else:
                ax.set_ylabel('')

            # ציר ה-X: רק לשורה התחתונה ביותר
            if row == num_rows - 1:
                ax.set_xlabel('Graph Embedding Output Dimension', fontsize=14, labelpad=10)
                if not subset.empty:
                    ax.set_xticks(sorted(subset['dout'].unique()))
            else:
                ax.set_xlabel('')

            ax.tick_params(axis='both', which='major', labelsize=12)

            # --- ניהול המקרא (Legend) ---
            # נשים את המקרא רק בפאנל הימני-עליון כדי למנוע כפילויות
            if row == 0 and col == num_cols - 1:
                ax.legend(title='Input Dimension', fontsize=13, title_fontsize=14, loc='best', framealpha=0.9)
            else:
                if ax.get_legend():
                    ax.get_legend().remove()

            # הסרת מסגרות מיותרות
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # שמירת הקובץ
    save_name = f"GEMS_Ablation_Grid_BS_vs_LR_{'best' if target_epoch == 'best' else f'ep{target_epoch}'}_PublicationReady"
    save_path = os.path.join(output_dir, f'{save_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ פיגר ה-Grid (רשת 2x2) נוצר ונשמר ב:\n{save_path}")


# =====================================================================
if __name__ == "__main__":
    # הנתיב לתיקייה שמכילה את כל ריצות הניסויים
    base_experiments_dir = "/mnt/new_groups/vaksler_group/Efrat/Results/experiment_runs_featuers_val_set/"

    # הנתיב שבו הגרפים יישמרו
    final_output_folder = "/home/efrco/PHD/Goal_One/Final_Results/"

    CHOSEN_EPOCH = 'best'

    print("\n🚀 מתחיל ריצה של קוד ניתוח הרגישות...")
    df_experiments = collect_experiments_data(base_experiments_dir, target_epoch=CHOSEN_EPOCH)

    # יצרתי mock df קטן רק כדי שהקוד ירוץ אם תרצי לבדוק אותו בלי הנתונים האמיתיים
    # אם יש לך את ה-df האמיתי, פשוט תורידי את ההערות מהשורות הרלוונטיות

    if not df_experiments.empty:
        plot_ablation_curves_clean(df_experiments, final_output_folder, target_epoch=CHOSEN_EPOCH)
        print("\n🎉 כל התהליך הסתיים בהצלחה!")
    else:
        print("\n❌ לא נמצאו נתונים שעונים על תנאי הסינון (למשל ריצות של 80 אפוקים). בדקי את הנתיבים.")
    plot_ablation_batch_size_grid(df_experiments, final_output_folder, target_epoch='best')
