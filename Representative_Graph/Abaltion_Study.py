import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# ==========================================
# 1. פונקציה לאיסוף הנתונים מכל התיקיות
# ==========================================
def collect_experiments_data(base_dir, target_epoch='best'):
    base_path = Path(base_dir)
    results_list = []

    # תבנית לאיתור הפרמטרים משם התיקייה
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

        # ==========================================
        # התיקון: המרה ל-int וסינון מדויק ל-80
        # ==========================================
        run_epochs_int = int(params["epochs"])

        if run_epochs_int != 80:
            # מדפיס כדי שתראי שהקוד באמת מדלג על מה שלא 80
            # print(f"  [מדלג] התיקייה {run_dir.name} מכילה {run_epochs_int} אפוקים (ולא 80).")
            continue

        # נתיבים לקבצי התוצאות
        results_folder = run_dir / "Netural" / "Results"
        if not results_folder.exists():
            results_folder = run_dir / "Results"
            if not results_folder.exists():
                continue

        # קריאת מודל ה-Combine (המודל המרכזי)
        combine_file = results_folder / "evaluation_summary_models_xgbs_combine_embedding.csv"
        without_file = results_folder / "evaluation_summary_models_xgbs_without_embedding.csv"
        only_file = results_folder / "evaluation_summary_models_xgbs_only_embedding.csv"

        try:
            if combine_file.exists():
                df_combine = pd.read_csv(combine_file)
                params["auc_combine"] = df_combine['auc'].mean()
                params["f1_combine"] = df_combine['f1_score'].mean()

            if without_file.exists():
                df_without = pd.read_csv(without_file)
                params["auc_without"] = df_without['auc'].mean()
                params["f1_without"] = df_without['f1_score'].mean()

            if only_file.exists():
                df_only = pd.read_csv(only_file)
                params["auc_only"] = df_only['auc'].mean()
                params["f1_only"] = df_only['f1_score'].mean()

            results_list.append(params)
        except Exception as e:
            print(f"⚠️ שגיאה בקריאת תיקייה {run_dir.name}: {e}")

    df_all = pd.DataFrame(results_list)
    print(f"✅ נאספו נתונים מ-{len(df_all)} ריצות שונות (רק של 80 אפוקים).")
    return df_all

def plot_modality_ablation(df, output_dir):
    """ פיגר 1: השוואת מקורות המידע (Modality) עבור הקונפיגורציה המנצחת """
    # מציאת הקונפיגורציה עם ה-AUC הגבוה ביותר במודל המשולב
    best_row = df.loc[df['auc_combine'].idxmax()]

    modalities = ['GEMS (Combined)', 'Classical Features Only', 'Graph Embeddings Only']
    aucs = [best_row['auc_combine'], best_row['auc_without'], best_row['auc_only']]

    plt.figure(figsize=(10, 6))
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e']  # ירוק ל-GEMS, כחול לפיצ'רים, כתום לגרף

    bars = plt.bar(modalities, aucs, color=colors, edgecolor='black', linewidth=1.5, width=0.6)

    plt.ylim(min(aucs) - 0.05, max(aucs) + 0.05)
    plt.ylabel('Mean Test AUC', fontsize=14)
    plt.title(
        f"Ablation Study: Information Modality Importance\n(Optimal Config: din={int(best_row['din'])}, dout={int(best_row['dout'])}, lr={best_row['lr']})",
        fontsize=16, fontweight='bold', pad=15)

    # הוספת הערכים המספריים על העמודות
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.005, f"{yval:.3f}", ha='center', va='bottom', fontsize=12,
                 fontweight='bold')

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    save_path = os.path.join(output_dir, 'Fig_1_Modality_Ablation.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ פיגר 1 (Modality Ablation) נוצר.")


def plot_bottleneck_effect(df, output_dir):
    """ פיגר 2: השפעת ה-dout (צוואר הבקבוק) על הביצועים """
    # נקבע פרמטרים שיהיו קבועים כדי לראות נקי את השפעת dout
    best_din = df.loc[df['auc_combine'].idxmax()]['din']
    best_lr = df.loc[df['auc_combine'].idxmax()]['lr']

    subset = df[(df['din'] == best_din) & (df['lr'] == best_lr)].copy()
    subset = subset.sort_values(by='dout')

    if len(subset) < 2:
        print("⚠️ אין מספיק נקודות ליצירת גרף Bottleneck (נדרשים ערכי dout שונים).")
        return

    plt.figure(figsize=(9, 6))
    plt.plot(subset['dout'], subset['auc_combine'], marker='o', markersize=10, linewidth=3, color='#800020',
             label='GEMS Test AUC')

    plt.xlabel('Graph Embedding Output Dimension (dout)', fontsize=14)
    plt.ylabel('Mean Test AUC', fontsize=14)
    plt.title(f"The Information Bottleneck Effect\n(Fixed: din={int(best_din)}, lr={best_lr})", fontsize=16,
              fontweight='bold', pad=15)

    # הדגשת הנקודה המקסימלית
    max_idx = subset['auc_combine'].idxmax()
    max_dout = subset.loc[max_idx, 'dout']
    max_auc = subset.loc[max_idx, 'auc_combine']
    plt.scatter([max_dout], [max_auc], color='gold', s=200, edgecolors='black', zorder=5,
                label=f'Optimal ({int(max_dout)})')

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=12, frameon=False)

    save_path = os.path.join(output_dir, 'Fig_2_Bottleneck_Effect.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ פיגר 2 (Bottleneck Effect) נוצר.")


def plot_architecture_heatmap(df, output_dir):
    """ פיגר 3: מפת חום של din לעומת dout """
    # נסנן ריצות פגומות וניקח את ה-AUC הטוב ביותר לכל שילוב של din ו-dout
    pivot_df = df.groupby(['din', 'dout'])['auc_combine'].max().reset_index()
    heatmap_data = pivot_df.pivot(index='din', columns='dout', values='auc_combine')

    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={'label': 'Mean Test AUC'},
                linewidths=1, linecolor='white', annot_kws={"size": 12})

    plt.xlabel('Output Embedding Dimension (dout)', fontsize=14)
    plt.ylabel('Input Vector Dimension (din)', fontsize=14)
    plt.title("Architecture Optimization Landscape\n(Heatmap of AUC)", fontsize=16, fontweight='bold', pad=15)

    # תיקון הצגת המספרים בצירים שיופיעו כשלמים
    plt.xticks(np.arange(len(heatmap_data.columns)) + 0.5, [int(c) for c in heatmap_data.columns])
    plt.yticks(np.arange(len(heatmap_data.index)) + 0.5, [int(r) for r in heatmap_data.index], rotation=0)

    save_path = os.path.join(output_dir, 'Fig_3_Architecture_Heatmap.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ פיגר 3 (Architecture Heatmap) נוצר.")


def plot_lr_robustness(df, output_dir):
    """ פיגר 4: יציבות קצב למידה """
    plt.figure(figsize=(8, 6))

    sns.boxplot(x='lr', y='auc_combine', data=df, palette="Set2", width=0.5, boxprops=dict(alpha=0.8))
    sns.stripplot(x='lr', y='auc_combine', data=df, color='black', alpha=0.5, jitter=True)

    plt.xlabel('Learning Rate', fontsize=14)
    plt.ylabel('Mean Test AUC', fontsize=14)
    plt.title("Model Robustness to Learning Rate", fontsize=16, fontweight='bold', pad=15)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    save_path = os.path.join(output_dir, 'Fig_4_LR_Robustness.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ פיגר 4 (LR Robustness) נוצר.")


# ==========================================
# 3. הפעלת הכל יחד
# ==========================================
base_experiments_dir = "/mnt/new_groups/vaksler_group/Efrat/Results/experiment_runs_featuers_new_try/"
final_output_folder = "/home/efrco/PHD/Goal_One/Final_Results/"

# יצירת התיקייה לגרפים אם לא קיימת
os.makedirs(final_output_folder, exist_ok=True)

# הרצה
print("מתחיל ביצירת פיגרים לפרק ה-Ablation Study...")
df_experiments = collect_experiments_data(base_experiments_dir)

if not df_experiments.empty:
    plot_modality_ablation(df_experiments, final_output_folder)
    plot_bottleneck_effect(df_experiments, final_output_folder)
    plot_architecture_heatmap(df_experiments, final_output_folder)
    plot_lr_robustness(df_experiments, final_output_folder)
    print("\n🎉 סיימנו! כל הגרפים נשמרו בהצלחה בתיקיית התוצאות הסופיות.")
else:
    print("❌ לא נאספו נתונים. נא לוודא שהנתיב לתיקיות הניסויים נכון ושקיימים קבצי CSV בתוכן.")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_bottleneck_boxplot(df, output_dir):
    """ פיגר 2 חלופי: השפעת ה-dout (צוואר הבקבוק) בתצוגת Boxplot נקייה """

    # נקבע פרמטרים שיהיו קבועים כדי לראות נקי את השפעת dout
    best_din = df.loc[df['auc_combine'].idxmax()]['din']
    best_lr = df.loc[df['auc_combine'].idxmax()]['lr']

    subset = df[(df['din'] == best_din) & (df['lr'] == best_lr)].copy()

    # המרה למספרים שלמים כדי שזה ייראה יפה בציר
    subset['dout'] = subset['dout'].astype(int)
    subset = subset.sort_values(by='dout')

    if len(subset) < 2:
        print("⚠️ אין מספיק נקודות ליצירת גרף Bottleneck.")
        return

    # הגדרות עיצוב מודרניות ופשוטות
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.figure(figsize=(9, 6))

    # יצירת קופסאות עדינות (מראות ממוצע, חציון ושונות)
    sns.boxplot(
        x='dout',
        y='auc_combine',
        data=subset,
        width=0.4,
        color='#c6dbef',  # תכלת עדין ונעים לעין
        boxprops=dict(edgecolor='black', linewidth=1.5, alpha=0.8),
        medianprops=dict(color='red', linewidth=2),
        whiskerprops=dict(color='black', linewidth=1.5),
        capprops=dict(color='black', linewidth=1.5),
        showfliers=False  # נסתיר חריגים מהקופסה כי אנחנו מציירים את כל הנקודות במילא
    )

    # הוספת הנקודות האמיתיות של הניסויים מעל הקופסאות
    sns.stripplot(
        x='dout',
        y='auc_combine',
        data=subset,
        color='#08306b',  # כחול כהה לנקודות
        alpha=0.7,
        jitter=0.15,  # מפזר קצת את הנקודות לצדדים כדי שלא יעלו אחת על השנייה
        size=8,
        linewidth=1,
        edgecolor='white'
    )

    # עיצוב טקסטים
    plt.xlabel('Graph Embedding Output Dimension (dout)', fontsize=14)
    plt.ylabel('Mean Test AUC', fontsize=14)
    plt.title(f"The Information Bottleneck Effect\n(Fixed: din={int(best_din)}, lr={best_lr})",
              fontsize=16, fontweight='bold', pad=15)

    # ניקיון הגרף (הסרת מסגרות עליונה וימנית)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # רשת רק בציר Y כדי להקל על קריאת הערכים
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # שמירה
    save_path = os.path.join(output_dir, 'Fig_2_Bottleneck_Boxplot.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'Fig_2_Bottleneck_Boxplot.pdf'), bbox_inches='tight')
    plt.show()
    print(f"✅ פיגר ה-Bottleneck החלופי נוצר ונשמר ב: {save_path}")


# ================================
# הפעלה לדוגמה:
# (בהנחה ש-df_experiments כבר קיים מהסקריפט הקודם)
# ================================
final_output_folder = "/home/efrco/PHD/Goal_One/Final_Results/"
# plot_bottleneck_boxplot(df_experiments, final_output_folder)


import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ==========================================
# 1. פונקציה לאיסוף הנתונים מכל התיקיות
# ==========================================
def collect_experiments_data(base_dir):
    base_path = Path(base_dir)
    results_list = []

    # תבנית לאיתור הפרמטרים משם התיקייה
    pattern = re.compile(r"run_epochs_(?P<epochs>[\d\.]+)_lr_(?P<lr>[\d\.]+)_din_(?P<din>[\d\.]+)_dout_(?P<dout>[\d\.]+)_bs_(?P<bs>[\d\.]+)")

    print("🔍 סורק את תיקיות הניסויים וקורא נתונים (מחשב מקסימום של ממוצע על פני פולדים)...")

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

        # ניגשים לקובץ התוצאות של המודל המשולב
        results_folder = run_dir / "Netural" / "Results"
        if not results_folder.exists():
            results_folder = run_dir / "Results" # גיבוי למקרה שהנתיב שונה
            if not results_folder.exists():
                continue

        combine_file = results_folder / "evaluation_summary_models_xgbs_combine_embedding.csv"

        try:
            if combine_file.exists():
                df_combine = pd.read_csv(combine_file)

                # --- התיקון המדעי הקריטי ---
                # 1. חישוב ממוצע של 10 הפולדים לכל אפוק בנפרד
                mean_per_epoch_combine = df_combine.groupby('epoch')[['auc', 'f1_score']].mean()

                # 2. לקיחת ערך ה-AUC הגבוה ביותר מבין כל האפוקים של הניסוי הזה
                params["auc_combine"] = mean_per_epoch_combine['auc'].max()
                params["f1_combine"] = mean_per_epoch_combine['f1_score'].max()

                results_list.append(params)
        except Exception as e:
            print(f"⚠️ שגיאה בקריאת תיקייה {run_dir.name}: {e}")

    df_all = pd.DataFrame(results_list)
    print(f"✅ נאספו נתונים מ-{len(df_all)} ריצות שונות.")
    return df_all

# ==========================================
# 2. פונקציה לייצור הגרפים
# ==========================================

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# ==========================================
# 1. פונקציה לאיסוף הנתונים מכל התיקיות
# ==========================================
def collect_experiments_data(base_dir, target_epoch='best'):
    """
    סורק את תיקיות הניסויים וקורא נתונים.
    target_epoch יכול להיות 'best' (לקיחת האפוק המקסימלי לכל ניסוי)
    או מספר שלם (למשל 68) כדי להשוות את כל הניסויים באותו אפוק.
    """
    base_path = Path(base_dir)
    results_list = []

    # תבנית לאיתור הפרמטרים משם התיקייה
    pattern = re.compile(
        r"run_epochs_(?P<epochs>[\d\.]+)_lr_(?P<lr>[\d\.]+)_din_(?P<din>[\d\.]+)_dout_(?P<dout>[\d\.]+)_bs_(?P<bs>[\d\.]+)")

    mode_text = "אפוק מקסימלי (Best)" if target_epoch == 'best' else f"אפוק ספציפי ({target_epoch})"
    print(f"🔍 סורק את תיקיות הניסויים וקורא נתונים (שיטת חישוב: {mode_text})...")

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

        results_folder = run_dir / "Netural" / "Results"
        if not results_folder.exists():
            results_folder = run_dir / "Results"
            if not results_folder.exists():
                continue

        combine_file = results_folder / "evaluation_summary_models_xgbs_combine_embedding.csv"

        try:
            if combine_file.exists():
                df_combine = pd.read_csv(combine_file)

                # חישוב ממוצע של 10 הפולדים לכל אפוק בנפרד
                mean_per_epoch = df_combine.groupby('epoch')[['auc', 'f1_score']].mean()

                if target_epoch == 'best':
                    # לקיחת הערך הגבוה ביותר מבין כל האפוקים
                    params["auc_combine"] = mean_per_epoch['auc'].max()
                    params["f1_combine"] = mean_per_epoch['f1_score'].max()
                else:
                    # משיכת הנתונים לאפוק הספציפי שביקשת
                    if target_epoch in mean_per_epoch.index:
                        params["auc_combine"] = mean_per_epoch.loc[target_epoch, 'auc']
                        params["f1_combine"] = mean_per_epoch.loc[target_epoch, 'f1_score']
                    else:
                        print(f"⚠️ דילוג: אפוק {target_epoch} לא קיים בניסוי {run_dir.name}.")
                        continue  # מדלגים על הניסוי הזה כי האפוק המבוקש לא קיים בו

                results_list.append(params)
        except Exception as e:
            print(f"⚠️ שגיאה בקריאת תיקייה {run_dir.name}: {e}")

    df_all = pd.DataFrame(results_list)
    print(f"✅ נאספו נתונים מ-{len(df_all)} ריצות שונות.")
    return df_all


# ==========================================
# 2. פונקציה לייצור הגרפים
# ==========================================
def plot_ablation_curves_clean(df, output_dir, target_epoch='best'):
    if df.empty:
        print("❌ אין נתונים להציג.")
        return

    # הגדרות עיצוב
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    sns.set_theme(style="whitegrid")

    # קביעת כיתוב הצירים לפי השיטה שנבחרה
    y_label_text = 'Max Mean Test AUC' if target_epoch == 'best' else f'Mean Test AUC (Epoch {target_epoch})'
    title_suffix = '(Best Epoch per Config)' if target_epoch == 'best' else f'(Evaluated at Epoch {target_epoch})'

    # ==========================================
    # פיגר 1: AUC כפונקציה של dout, צבע = din
    # ==========================================
    best_lr = df.loc[df['auc_combine'].idxmax()]['lr']
    subset_lr = df[df['lr'] == best_lr].copy()

    subset_lr['dout'] = subset_lr['dout'].astype(int)
    subset_lr['din'] = subset_lr['din'].astype(int)
    subset_lr = subset_lr.sort_values(by=['din', 'dout'])

    plt.figure(figsize=(9, 6))
    sns.lineplot(
        data=subset_lr, x='dout', y='auc_combine', hue='din',
        marker='o', markersize=9, linewidth=2.5, palette='Set1'
    )

    plt.title(f"Ablation: Input vs. Output Dimension\n{title_suffix} | Fixed LR: {best_lr}",
              fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Graph Embedding Output Dimension (dout)', fontsize=14)
    plt.ylabel(y_label_text, fontsize=14)

    plt.legend(title='Input Dimension (din)', fontsize=12, title_fontsize=13, loc='best', framealpha=0.9)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xticks(sorted(subset_lr['dout'].unique()))

    save_name_1 = f"GEMS_Ablation_dout_vs_din_{'best' if target_epoch == 'best' else f'ep{target_epoch}'}"
    save_path_1 = os.path.join(output_dir, f'{save_name_1}.png')
    plt.savefig(save_path_1, dpi=300, bbox_inches='tight')
    plt.show()

    # ==========================================
    # פיגר 2: AUC כפונקציה של dout, צבע = lr
    # ==========================================
    best_din = df.loc[df['auc_combine'].idxmax()]['din']
    subset_din = df[df['din'] == best_din].copy()

    subset_din['dout'] = subset_din['dout'].astype(int)
    subset_din = subset_din.sort_values(by=['lr', 'dout'])

    plt.figure(figsize=(9, 6))
    sns.lineplot(
        data=subset_din, x='dout', y='auc_combine', hue='lr',
        marker='s', markersize=9, linewidth=2.5, palette='Dark2'
    )

    plt.title(f"Ablation: Output Dimension vs. Learning Rate\n{title_suffix} | Fixed din: {int(best_din)}",
              fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Graph Embedding Output Dimension (dout)', fontsize=14)
    plt.ylabel(y_label_text, fontsize=14)

    plt.legend(title='Learning Rate (lr)', fontsize=12, title_fontsize=13, loc='best', framealpha=0.9)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xticks(sorted(subset_din['dout'].unique()))

    save_name_2 = f"GEMS_Ablation_dout_vs_lr_{'best' if target_epoch == 'best' else f'ep{target_epoch}'}"
    save_path_2 = os.path.join(output_dir, f'{save_name_2}.png')
    plt.savefig(save_path_2, dpi=300, bbox_inches='tight')
    plt.show()


# ==========================================
# 3. בלוק ההפעלה המרכזי
# ==========================================
if __name__ == "__main__":
    base_experiments_dir = "/mnt/new_groups/vaksler_group/Efrat/Results/experiment_runs_featuers_new_try/"
    final_output_folder = "/home/efrco/PHD/Goal_One/Final_Results/"
    os.makedirs(final_output_folder, exist_ok=True)

    # ---------------------------------------------------------
    # הגדרת אופן ההרצה:
    # שנית כאן ל-'best' כדי לקבל מקסימום, או למספר (למשל 68)
    # ---------------------------------------------------------
    CHOSEN_EPOCH = 'best'

    df_experiments = collect_experiments_data(base_experiments_dir, target_epoch=CHOSEN_EPOCH)
    plot_ablation_curves_clean(df_experiments, final_output_folder, target_epoch=CHOSEN_EPOCH)
    print("🎉 כל הגרפים נוצרו בהצלחה!")

import os
import re
import pandas as pd
from pathlib import Path

import os
import re
import pandas as pd
from pathlib import Path


def find_best_epoch_per_configuration(base_dir, output_dir):
    """
    סורק את כל הניסויים, מחשב ממוצע פולדים לכל אפוק,
    ומוצא את האפוק שנותן את ה-AUC הממוצע הגבוה ביותר לכל קונפיגורציה.
    """
    base_path = Path(base_dir)
    results_list = []

    pattern = re.compile(
        r"run_epochs_(?P<epochs>[\d\.]+)_lr_(?P<lr>[\d\.]+)_din_(?P<din>[\d\.]+)_dout_(?P<dout>[\d\.]+)_bs_(?P<bs>[\d\.]+)")

    print("🔍 מתחיל לסרוק את הניסויים ולחשב ממוצעים...")

    for run_dir in base_path.glob("run_*"):
        if not run_dir.is_dir():
            continue

        match = pattern.search(run_dir.name)
        if not match:
            continue

        results_folder = run_dir / "Netural" / "Results"
        if not results_folder.exists():
            results_folder = run_dir / "Results"
            if not results_folder.exists():
                continue

        combine_file = results_folder / "evaluation_summary_models_xgbs_combine_embedding.csv"

        try:
            if combine_file.exists():
                df_combine = pd.read_csv(combine_file)

                mean_per_epoch = df_combine.groupby('epoch')['auc'].mean()

                best_epoch = mean_per_epoch.idxmax()
                max_mean_auc = mean_per_epoch.max()

                results_list.append({
                    "Learning_Rate": float(match.group("lr")),
                    "Input_Dim_(din)": int(float(match.group("din"))),
                    "Output_Dim_(dout)": int(float(match.group("dout"))),
                    "Batch_Size": int(float(match.group("bs"))),
                    "Best_Epoch": int(best_epoch),
                    "Max_Mean_AUC": max_mean_auc,
                    "Configuration_Folder": run_dir.name
                })

        except Exception as e:
            print(f"⚠️ שגיאה בקריאת תיקייה {run_dir.name}: {e}")

    df_best_epochs = pd.DataFrame(results_list)

    if df_best_epochs.empty:
        print("❌ לא נמצאו נתונים לעיבוד.")
        return

    # ==========================================
    # התיקון: עיגול כל המספרים העשרוניים בטבלה ל-4 ספרות
    # ==========================================
    df_best_epochs = df_best_epochs.round(3)

    df_best_epochs = df_best_epochs.sort_values(by="Max_Mean_AUC", ascending=False).reset_index(drop=True)

    print("\n🏆 סיכום: האפוק המנצח וה-AUC המקסימלי לכל קונפיגורציה 🏆")
    print("=" * 85)
    columns_to_print = ["Learning_Rate", "Input_Dim_(din)", "Output_Dim_(dout)", "Best_Epoch", "Max_Mean_AUC"]
    print(df_best_epochs[columns_to_print].to_string(index=False))
    print("=" * 85)

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "GEMS_Best_Epochs_Summary.csv")
    df_best_epochs.to_csv(save_path, index=False)

    print(f"\n✅ קובץ הסיכום נשמר בהצלחה בנתיב:")
    print(f"   {save_path}")


# ================================
# הפעלה:
# ================================
base_experiments_dir = "/mnt/new_groups/vaksler_group/Efrat/Results/experiment_runs_featuers_new_try/"
final_output_folder = "/home/efrco/PHD/Goal_One/Final_Results/"

find_best_epoch_per_configuration(base_experiments_dir, final_output_folder)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_ablation_curves_clean(df, output_dir, target_epoch='best'):
    if df.empty:
        print("❌ אין נתונים להציג.")
        return

    # הגדרות עיצוב
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    sns.set_theme(style="whitegrid")

    y_label_text = 'Max Mean Test AUC' if target_epoch == 'best' else f'Mean Test AUC (Epoch {target_epoch})'
    title_suffix = '(Best Epoch per Config)' if target_epoch == 'best' else f'(Evaluated at Epoch {target_epoch})'

    # ==========================================
    # פיגר 1 (משודרג): AUC כפונקציה של dout, צבע = din
    # מחולק לפאנלים לפי קצב הלמידה (Learning Rate)
    # ==========================================
    unique_lrs = sorted(df['lr'].unique())
    num_lrs = len(unique_lrs)

    # יצירת פיגר מפוצל: שורה אחת, מספר עמודות כמספר קצבי הלמידה
    fig, axes = plt.subplots(1, num_lrs, figsize=(7 * num_lrs, 6), sharey=True)

    # אם יש רק lr אחד, הופכים את axes לרשימה כדי שהלולאה תעבוד
    if num_lrs == 1:
        axes = [axes]

    for i, current_lr in enumerate(unique_lrs):
        ax = axes[i]

        # סינון הנתונים רק לקצב הלמידה הנוכחי בלולאה
        subset_lr = df[df['lr'] == current_lr].copy()
        subset_lr['dout'] = subset_lr['dout'].astype(int)
        subset_lr['din'] = subset_lr['din'].astype(int)
        subset_lr = subset_lr.sort_values(by=['din', 'dout'])

        # ציור העקומות בתוך הפאנל הספציפי (ax=ax)
        sns.lineplot(
            data=subset_lr, x='dout', y='auc_combine', hue='din',
            marker='o', markersize=9, linewidth=2.5, palette='Set1', ax=ax
        )

        ax.set_title(f"Learning Rate: {current_lr}", fontsize=14, fontweight='bold')
        ax.set_xlabel('Graph Embedding Output Dimension (dout)', fontsize=13)

        # כיתוב ציר Y רק לפאנל השמאלי ביותר
        if i == 0:
            ax.set_ylabel(y_label_text, fontsize=14)
        else:
            ax.set_ylabel('')

        # סידור מקרא - נציג אותו רק בפאנל הימני ביותר כדי לא להעמיס
        if i == num_lrs - 1:
            ax.legend(title='Input Dimension (din)', fontsize=12, title_fontsize=13, loc='best', framealpha=0.9)
        else:
            if ax.get_legend():
                ax.get_legend().remove()

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks(sorted(subset_lr['dout'].unique()))

    # כותרת ראשית מעל כל הפאנלים
    fig.suptitle(f"Ablation: Input vs. Output Dimension across Learning Rates\n{title_suffix}",
                 fontsize=16, fontweight='bold', y=1.05)

    plt.tight_layout()

    save_name_1 = f"GEMS_Ablation_dout_vs_din_multi_LR_{'best' if target_epoch == 'best' else f'ep{target_epoch}'}"
    save_path_1 = os.path.join(output_dir, f'{save_name_1}.png')
    plt.savefig(save_path_1, dpi=300, bbox_inches='tight')
    plt.savefig(save_path_1.replace('.png', '.pdf'), bbox_inches='tight')
    plt.show()
    print(f"✅ פיגר 1 (מרובה פאנלים) נוצר ונשמר ב: {save_path_1}")

    # ==========================================
    # פיגר 2: AUC כפונקציה של dout, צבע = lr (עבור din קבוע)
    # נשאר כפי שהיה, להשלמת התמונה
    # ==========================================
    best_din = df.loc[df['auc_combine'].idxmax()]['din']
    subset_din = df[df['din'] == best_din].copy()

    subset_din['dout'] = subset_din['dout'].astype(int)
    subset_din = subset_din.sort_values(by=['lr', 'dout'])

    plt.figure(figsize=(9, 6))
    sns.lineplot(
        data=subset_din, x='dout', y='auc_combine', hue='lr',
        marker='s', markersize=9, linewidth=2.5, palette='Dark2'
    )

    plt.title(f"Ablation: Output Dimension vs. Learning Rate\n{title_suffix} | Fixed din: {int(best_din)}",
              fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Graph Embedding Output Dimension (dout)', fontsize=14)
    plt.ylabel(y_label_text, fontsize=14)

    plt.legend(title='Learning Rate (lr)', fontsize=12, title_fontsize=13, loc='best', framealpha=0.9)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xticks(sorted(subset_din['dout'].unique()))

    save_name_2 = f"GEMS_Ablation_dout_vs_lr_{'best' if target_epoch == 'best' else f'ep{target_epoch}'}"
    save_path_2 = os.path.join(output_dir, f'{save_name_2}.png')
    plt.savefig(save_path_2, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ פיגר 2 נוצר ונשמר ב: {save_path_2}")

if __name__ == "__main__":
    base_experiments_dir = "/mnt/new_groups/vaksler_group/Efrat/Results/experiment_runs_featuers_new_try/"
    final_output_folder = "/home/efrco/PHD/Goal_One/Final_Results/"
    os.makedirs(final_output_folder, exist_ok=True)

    # ---------------------------------------------------------
    # הגדרת אופן ההרצה:
    # שנית כאן ל-'best' כדי לקבל מקסימום, או למספר (למשל 68)
    # ---------------------------------------------------------
    CHOSEN_EPOCH = 'best'

    df_experiments = collect_experiments_data(base_experiments_dir, target_epoch=CHOSEN_EPOCH)
    plot_ablation_curves_clean(df_experiments, final_output_folder, target_epoch=CHOSEN_EPOCH)
    print("🎉 כל הגרפים נוצרו בהצלחה!")
