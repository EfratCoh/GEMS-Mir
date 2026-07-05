import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import pandas as pd
import glob

##################################################################################################################
############################################Find all configuration summary ######################3######3
import os
import pandas as pd
import glob

import os
import pandas as pd
import glob

import os
import pandas as pd
import glob


def summarize_gnn_best_epochs(base_dir):
    print(f"🔍 סורק תיקיות ניסויים בנתיב: {base_dir}")
    summary_data = []

    # חיפוש כל תיקיות הריצה (run_*)
    experiment_dirs = glob.glob(os.path.join(base_dir, "run_*"))

    for exp_dir in experiment_dirs:
        config_name = os.path.basename(exp_dir)

        # טיפול בשם התיקייה (אות גדולה/קטנה)
        embedding_dir = os.path.join(exp_dir, "Data_embedding")
        if not os.path.exists(embedding_dir):
            embedding_dir = os.path.join(exp_dir, "Data_Embedding")

        # מציאת כל קבצי הסיכום של הפולדים
        metrics_files = glob.glob(os.path.join(embedding_dir, "fold*", "training_metrics_comprehensive_fold*.csv"))

        if not metrics_files:
            print(f"⚠️ לא נמצאו קבצי training_metrics בתיקייה {config_name}. מדלג.")
            continue

        try:
            all_folds_metrics = []
            for m_file in metrics_files:
                df = pd.read_csv(m_file)
                all_folds_metrics.append(df)

            # איחוד הנתונים מכל הפולדים
            combined_metrics = pd.concat(all_folds_metrics, ignore_index=True)

            # חישוב ממוצע Val AUC לכל אפוק
            mean_per_epoch = combined_metrics.groupby('epoch')['val_auc'].mean().reset_index()

            # מציאת האפוק המנצח לפי ה-Val AUC המקסימלי
            best_idx = mean_per_epoch['val_auc'].idxmax()
            best_epoch = int(mean_per_epoch.loc[best_idx, 'epoch'])
            best_val_auc = mean_per_epoch.loc[best_idx, 'val_auc']

            summary_data.append({
                'Configuration': config_name,
                'Best_Epoch': best_epoch,
                'Val_AUC': best_val_auc
            })

        except Exception as e:
            print(f"❌ שגיאה בעיבוד התיקייה {config_name}: {e}")

    if summary_data:
        summary_df = pd.DataFrame(summary_data)

        # עיגול ל-3 ספרות אחרי הנקודה
        summary_df['Val_AUC'] = summary_df['Val_AUC'].round(3)

        # מיון מהתוצאה הגבוהה לנמוכה
        summary_df = summary_df.sort_values(by='Val_AUC', ascending=False).reset_index(drop=True)

        # שמירת הקובץ
        output_file = os.path.join(base_dir, "GNN_Validation_Best_Epochs.csv")
        summary_df.to_csv(output_file, index=False)

        # ==============================================================
        # הדפסת 3 הקונפיגורציות המנצחות הכלליות למסך
        # ==============================================================
        print("\n" + "🏆" * 20)
        print("🌟 3 הקונפיגורציות המובילות הן:")

        medals = ["🥇", "🥈", "🥉"]
        num_to_print = min(3, len(summary_df))  # מוודא שלא נקרוס אם יש פחות מ-3 הרצות

        for i in range(num_to_print):
            row = summary_df.iloc[i]
            print(f"\n{medals[i]} מקום {i + 1}:")
            print(f"   ➤ שם: {row['Configuration']}")
            print(f"   ➤ באפוק: {row['Best_Epoch']}")
            print(f"   ➤ עם Val AUC של: {row['Val_AUC']}")

        print("\n" + "🏆" * 20 + "\n")

        print(f"✅ טבלת הסיכום המלאה נשמרה בנתיב:\n{output_file}")

        return summary_df
    else:
        print("\n⚠️ לא נמצאו נתונים מתאימים לסיכום.")
        return None


# =======================================================
if __name__ == "__main__":
    target_directory = "/mnt/new_groups/vaksler_group/Efrat/Results/experiment_runs_featuers_val_set"
    # summarize_gnn_best_epochs(target_directory)


input_csv = "/mnt/new_groups/vaksler_group/Efrat/Results/experiment_runs_featuers_val_set/run_epochs_80.0_lr_0.001_din_128.0_dout_256.0_bs_64.0/Results/gems_metrics_models_xgbs_combine_embedding.csv"
final_results_dir = '/home/efrco/PHD/Goal_One/Final_Results/PartA'

############################## Function 1 #############################################################
# Summary results and average for each epoch across fold
def generate_detailed_summary(file_path, output_folder):
    # 1. קריאת קובץ התוצאות המקורי
    df = pd.read_csv(file_path)

    metrics = ['accuracy', 'auc', 'f1_score']
    if 'auprc' in df.columns:
        metrics.append('auprc')

    summary_df = df.groupby('epoch')[metrics].agg(['mean', 'std'])

    summary_df.columns = [f"{col.upper()}_{stat.capitalize()}" for col, stat in summary_df.columns]
    summary_df = summary_df.reset_index()

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_path = os.path.join(output_folder, 'GEMS_Detailed_Metrics_Per_Epoch_early.csv')
    summary_df.to_csv(output_path, index=False)

    print(f"✅ קובץ הסיכום המפורט נוצר בהצלחה ונשמר בנתיב:\n{output_path}")

    return summary_df

# detailed_summary = generate_detailed_summary(input_csv, final_results_dir)

##########################Function 2#################################################
# this func find the best optimal epoch on the val set - only on the gnnn
def analyze_validation_results(results_dir):

    in_dir= Path("/mnt/new_groups/vaksler_group/Efrat/Results/experiment_runs_featuers_val_set")
    results_dir = final_results_dir
    results_path = Path(results_dir)

    # 1. קריאת כל קבצי ה-CSV של כל הפולדים
    all_files = list(in_dir.rglob("training_metrics_comprehensive_fold*.csv"))
    if not all_files:
        print("❌ לא נמצאו קבצי תוצאות בנתיב שסופק.")
        return

    df_list = [pd.read_csv(f) for f in all_files]
    df = pd.concat(df_list, ignore_index=True)

    # 2. קיבוץ לפי אפוק וחישוב ממוצע וסטיית תקן לכל מדד
    metrics = ['train_loss', 'val_loss', 'val_auc', 'test_loss', 'test_accuracy', 'test_auc', 'test_auprc', 'test_f1']
    epoch_stats = df.groupby('epoch')[metrics].agg(['mean', 'std'])

    # --- הוספה: שמירת הממוצעים לכל אפוק לקובץ CSV ---
    # נשטח את שמות העמודות כדי שהקובץ יהיה קריא (למשל: val_auc_mean במקום כותרת כפולה)
    epoch_stats_to_save = epoch_stats.copy()
    epoch_stats_to_save.columns = [f"{col[0]}_{col[1]}" for col in epoch_stats_to_save.columns]

    csv_save_path = os.path.join(results_dir, "Mean_Epoch_Metrics_Summary.csv")
    epoch_stats_to_save.to_csv(csv_save_path)
    print(f"✅ קובץ סיכום הממוצעים לכל אפוק נשמר בהצלחה ב:\n   {csv_save_path}")
    # ---------------------------------------------------

    # 3. מציאת האפוק המנצח על סמך *Validation AUC*
    best_epoch = epoch_stats[('val_auc', 'mean')].idxmax()
    best_stats = epoch_stats.loc[best_epoch]

    print("\n" + "=" * 50)
    print(f"🏆 האפוק המנצח (נבחר לפי Max Mean Val AUC): {best_epoch}")
    print("=" * 50)
    print("ביצועי המבחן האמיתיים (Test Set) באפוק זה:")
    print(f"📊 Test AUC:   {best_stats[('test_auc', 'mean')]:.4f} ± {best_stats[('test_auc', 'std')]:.4f}")
    print(f"📊 Test AUPRC: {best_stats[('test_auprc', 'mean')]:.4f} ± {best_stats[('test_auprc', 'std')]:.4f}")
    print(f"📊 Test F1:    {best_stats[('test_f1', 'mean')]:.4f} ± {best_stats[('test_f1', 'std')]:.4f}")
    print(f"📊 Test Acc:   {best_stats[('test_accuracy', 'mean')]:.4f} ± {best_stats[('test_accuracy', 'std')]:.4f}")
    print("=" * 50)

    # ==========================================
    # 4. ציור עקומת הלמידה (Validation vs. Test)
    # ==========================================
    # הגדרות עיצוב לפרסום
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']

    fig, ax1 = plt.subplots(figsize=(11, 6))

    epochs = epoch_stats.index
    val_auc_mean = epoch_stats[('val_auc', 'mean')]
    test_auc_mean = epoch_stats[('test_auc', 'mean')]
    train_loss_mean = epoch_stats[('train_loss', 'mean')]

    # --- ציר Y שמאלי - AUC למבחן וולידציה ---
    ax1.set_xlabel('Epoch', fontsize=13)
    ax1.set_ylabel('Mean AUC', fontsize=13)

    # קו ולידציה (כחול רציף)
    line1 = ax1.plot(epochs, val_auc_mean, color='#1f77b4', lw=2.5, label='Validation AUC')
    # קו טסט (ירוק מקווקו)
    line2 = ax1.plot(epochs, test_auc_mean, color='#2ca02c', lw=2.5, linestyle='--', alpha=0.9, label='Test AUC')

    ax1.tick_params(axis='y', labelsize=11)
    ax1.tick_params(axis='x', labelsize=11)

    # סימון האפוק המנצח
    line_best = ax1.axvline(x=best_epoch, color='#d62728', linestyle='-.', lw=1.5,
                            label=f'Optimal Epoch ({best_epoch})')

    # --- ציר Y ימני - Train Loss ---
    ax2 = ax1.twinx()
    ax2.set_ylabel('Train Loss', color='#ff7f0e', fontsize=13)
    line3 = ax2.plot(epochs, train_loss_mean, color='#ff7f0e', lw=2, linestyle=':', alpha=0.8, label='Train Loss')
    ax2.tick_params(axis='y', labelcolor='#ff7f0e', labelsize=11)

    # --- עיצוב מקרא וכותרות ---
    plt.title('GCNN Learning Dynamics: Validation vs. Test Performance', fontsize=15, fontweight='bold', pad=15)

    # איחוד כל הקווים למקרא אחד מסודר
    lines = line1 + line2 + line3 + [line_best]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', fontsize=11, frameon=True, edgecolor='gray')

    # הוספת רשת עדינה לציר השמאלי בלבד
    ax1.grid(True, linestyle='-', alpha=0.3, color='gray')

    fig.tight_layout()

    # --- שמירת הקובץ ---
    save_path = os.path.join(results_dir, "Learning_Curve_Val_vs_Test.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ עקומת הלמידה (Val & Test) נשמרה ב: {save_path}")

    return best_epoch

# optimal_epoch = analyze_validation_results(input_csv)
# print(f"Optimal Epoch: {optimal_epoch}")
target_epoch = 61



############################## Function 3 #############################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import os

# create ruc auc figure on the best epoch that find in func 2
def plot_publication_cv_curves(predictions_csv, target_epoch, output_dir):
    # 1. עיצוב גלובלי
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']

    # 2. קריאת הנתונים וסינון לאפוק הנבחר
    if not os.path.exists(predictions_csv):
        print(f"❌ לא נמצא קובץ הנתונים בנתיב: {predictions_csv}")
        return

    df = pd.read_csv(predictions_csv)

    # וידוא שסוג הנתונים תואם
    df['epoch'] = df['epoch'].astype(str)
    target_epoch = str(target_epoch)
    df_epoch = df[df['epoch'] == target_epoch]

    if df_epoch.empty:
        print(f"❌ לא נמצאו נתונים עבור אפוק {target_epoch}")
        return

    # 3. הכנת משתנים לאגירת הנתונים של 10 הפולדים
    mean_fpr = np.linspace(0, 1, 100)
    mean_recall = np.linspace(0, 1, 100)

    tprs = []
    aucs = []
    precisions = []
    auprcs = []

    folds = df_epoch['fold'].unique()

    # חישוב ה-Baseline של עקומת ה-PR
    baseline_chance = df_epoch['True_Label'].mean()

    # חישוב המדדים לכל פולד בנפרד
    for f in folds:
        fold_data = df_epoch[df_epoch['fold'] == f]
        y_true = fold_data['True_Label'].values
        y_prob = fold_data['Prob_Score'].values

        # חישוב ל-ROC
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(auc(fpr, tpr))

        # חישוב ל-PR
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        # הפיכת המערכים
        interp_precision = np.interp(mean_recall, recall[::-1], precision[::-1])
        precisions.append(interp_precision)
        auprcs.append(average_precision_score(y_true, y_prob))

    # ממוצעים וסטיות תקן לכל הפולדים
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    mean_precision = np.mean(precisions, axis=0)
    mean_auprc = np.mean(auprcs)
    std_auprc = np.std(auprcs)

    # --- Brand Colors ---
    GEM_GREEN_DARK = '#2E7D32'
    DEEP_PURPLE = '#6A1B9A'
    REFERENCE_RED = '#E53935'
    GREY_FOLDS = '#BDBDBD'

    # ==========================================
    # 4. ציור הגרפים (פאנל כפול)
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Panel A: ROC Curve ---
    for i in range(len(folds)):
        ax1.plot(mean_fpr, tprs[i], color=GREY_FOLDS, alpha=0.4, lw=1, zorder=1)

    ax1.plot(mean_fpr, mean_tpr, color=GEM_GREEN_DARK, lw=3,
             label=fr'Mean ROC ($\mathit{{AUC}}$ = {mean_auc:.3f} $\pm$ {std_auc:.3f})', zorder=5)
    ax1.plot([0, 1], [0, 1], linestyle='--', lw=2, color='#999999', alpha=0.9, label='Random Classifier (0.500)', zorder=4)
    #REFERENCE_RED
    ax1.set_xlabel('False Positive Rate', fontsize=14, labelpad=10)
    ax1.set_ylabel('True Positive Rate', fontsize=14, labelpad=10)
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.02])

    # --- Panel B: Precision-Recall Curve ---
    for i in range(len(folds)):
        ax2.plot(mean_recall, precisions[i], color=GREY_FOLDS, alpha=0.4, lw=1, zorder=1)

    ax2.plot(mean_recall, mean_precision, color=DEEP_PURPLE, lw=3,
             label=fr'Mean PR ($\mathit{{PR−AUC}}$ = {mean_auprc:.3f} $\pm$ {std_auprc:.3f})', zorder=5)
    ax2.axhline(y=baseline_chance, linestyle='--', lw=2, color='#999999', alpha=0.9,
                label=f'Positive Class Baseline ({baseline_chance:.3f})', zorder=4)

    ax2.set_xlabel('Recall', fontsize=14, labelpad=10)
    ax2.set_ylabel('Precision', fontsize=14, labelpad=10)
    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([-0.02, 1.02])

    # --- עיצוב אחיד לשני הפאנלים ---
    for ax, letter in zip([ax1, ax2], ['A', 'B']):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # גריד עדין שתואם לגרפים המרחביים
        ax.grid(True, linestyle='-', color='#EEEEEE', alpha=0.7, zorder=0)

        ax.tick_params(axis='both', which='major', labelsize=12)

        # עיצוב המקרא
        ax.legend(loc='lower right', fontsize=12, frameon=True, facecolor='white', edgecolor='#DDDDDD')

        # אותיות פאנל (A, B) בצד שמאל למעלה
        ax.text(-0.1, 1.05, letter, transform=ax.transAxes,
                fontsize=18, fontweight='bold', va='top', ha='right', color='#333333')

    plt.tight_layout(pad=3.0)

    # 5. יצירת תיקייה ושמירה
    os.makedirs(output_dir, exist_ok=True)
    save_prefix = os.path.join(output_dir, f'GEMS_ROC_PR_Curves_Epoch_{target_epoch}_Styled')

    plt.savefig(f'{save_prefix}.png', dpi=300, bbox_inches='tight')

    print("-" * 50)
    print(f"✅ הגרפים ברמת Publication (מעוצבים) נוצרו בהצלחה!")
    print(f"נתיב שמירה: {output_dir}")
    print(f"סטטיסטיקה (Epoch {target_epoch}):")
    print(f"   Mean AUC:   {mean_auc:.3f} ± {std_auc:.3f}")
    print(f"   Mean AUPRC: {mean_auprc:.3f} ± {std_auprc:.3f}")
    print("-" * 50)

    plt.close()

input_csv = "/mnt/new_groups/vaksler_group/Efrat/Results/experiment_runs_featuers_val_set/run_epochs_80.0_lr_0.001_din_128.0_dout_256.0_bs_64.0/Results/gems_predictions_models_xgbs_combine_embedding.csv"

# this creat figure on the final xgboost
# mean auc and the roc is also mean auc
plot_publication_cv_curves(predictions_csv=input_csv, target_epoch=target_epoch, output_dir=final_results_dir)


#################################SHAP############################################
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

def plot_all_folds_shap_for_epoch(shap_csv_path, target_epoch, top_n=10, output_dir=None):
    """
    מייצרת תרשים מרובה-פאנלים ל-SHAP.
    גרסה מוגדלת: פונטים גדולים וברורים יותר, עם קנבס ומרווחים מותאמים
    ששומרים על יישור שמאלי ומונעים חיתוך.
    """
    if not os.path.exists(shap_csv_path):
        print(f"❌ שגיאה: הקובץ לא נמצא בנתיב: {shap_csv_path}")
        return

    # 1. קריאת הנתונים
    df = pd.read_csv(shap_csv_path)
    shap_col = [col for col in df.columns if col.startswith('mean_abs')]
    if not shap_col:
        print("❌ שגיאה: לא נמצאה עמודת SHAP.")
        return
    shap_col_name = shap_col[0]

    df['epoch'] = df['epoch'].astype(str)
    df['fold'] = df['fold'].astype(str)

    target_epoch_str = str(target_epoch)
    epoch_data = df[df['epoch'] == target_epoch_str]

    if epoch_data.empty:
        print(f"❌ שגיאה: לא נמצאו נתונים עבור אפוק {target_epoch}.")
        return

    folds = sorted(epoch_data['fold'].unique(), key=lambda x: int(x) if x.isdigit() else x)

    if len(folds) == 0:
        print("❌ שגיאה: לא נמצאו פולדים בנתונים.")
        return

    global_max_shap = epoch_data.groupby('fold').apply(
        lambda x: x.sort_values(by=shap_col_name, ascending=False).head(top_n)[shap_col_name].max()
    ).max()

    # --- Official Brand Colors ---
    GEM_GREEN_LIGHT = '#C1E1C1'
    GEM_GREEN_DARK = '#2E7D32'
    MTI_LILAC = '#C8A2C8'
    MTI_EDGE = '#9966CC'

    # ==========================================
    # ציור הגרף (10 פאנלים)
    # ==========================================
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']

    n_cols = 5
    n_rows = int(np.ceil(len(folds) / n_cols))

    # הגדלנו את הקנבס ל-55x18 (עדיין 100% בטוח לשמירה, אבל נותן יותר אוויר לפונטים)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(55, 18), sharex=False)
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for idx, f in enumerate(folds):
        ax = axes[idx]

        fold_data = epoch_data[epoch_data['fold'] == f]
        feat_imp = fold_data.sort_values(by=shap_col_name, ascending=False).head(top_n).copy()

        feat_imp['display_name'] = feat_imp['feature']

        face_colors = [MTI_LILAC if str(feat).startswith('dim_') else GEM_GREEN_LIGHT for feat in feat_imp['feature']]
        edge_colors = [MTI_EDGE if str(feat).startswith('dim_') else GEM_GREEN_DARK for feat in feat_imp['feature']]

        sns.barplot(
            data=feat_imp,
            x=shap_col_name,
            y='display_name',
            palette=face_colors,
            ax=ax,
            linewidth=2.5,
            zorder=3
        )

        for i, patch in enumerate(ax.patches):
            if i < len(edge_colors):
                patch.set_edgecolor(edge_colors[i])

        # פונטים מוגדלים משמעותית לכל האלמנטים
        ax.set_title(f'Fold {f}', fontsize=32, fontweight='bold', color='#333333', pad=18)
        ax.set_xlabel(r'Mean |$\mathit{SHAP}$|' if idx >= len(folds) - n_cols else '', fontsize=28, labelpad=15)
        ax.set_ylabel('')

        ax.set_xlim(0, global_max_shap * 1.05)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))

        ax.grid(axis='x', linestyle='-', color='#EEEEEE', alpha=1.0, zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#DDDDDD')
        ax.spines['left'].set_color('#DDDDDD')

        ax.tick_params(axis='x', labelsize=22)
        ax.set_yticks(range(len(feat_imp)))
        ax.set_yticklabels([])

        # הטקסט של המאפיינים - הוגדל מ-18 ל-26!
        # הוזז למיקום -1.8 כדי לפנות את המקום הנוסף שהפונט הגדול דורש
        for i, name in enumerate(feat_imp['display_name']):
            ax.text(-1.85, i, name,
                    ha='left', va='center',
                    fontsize=26, color='#222222',
                    transform=ax.get_yaxis_transform())

    for idx in range(len(folds), len(axes)):
        fig.delaxes(axes[idx])

    # כותרת ראשית ומקרא מוגדלים בהתאמה
    fig.suptitle(fr'$\mathit{{SHAP}}$ Feature Importance Across Folds',
                 fontsize=42, fontweight='bold', y=1.05, color='#222222')

    legend_elements = [
        Patch(facecolor=GEM_GREEN_LIGHT, edgecolor=GEM_GREEN_DARK, linewidth=3.0, label='Handcrafted Features'),
        Patch(facecolor=MTI_LILAC, edgecolor=MTI_EDGE, linewidth=3.0, label='Embedding Features (dim_X)')
    ]

    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.00),
               ncol=2, fontsize=26, frameon=True, facecolor='white', edgecolor='#DDDDDD', borderpad=1.0)

    # המרווח (wspace) הוגדל ל-1.8 והשול השמאלי (left) ל-0.14 כדי להכיל את הטקסט המוגדל
    fig.subplots_adjust(left=0.14, right=0.96, wspace=1.8, hspace=0.45, top=0.90)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = '.'

    save_prefix = os.path.join(output_dir, f'GEMS_SHAP_10_Panels_Epoch_{target_epoch}_LargerText')

    plt.savefig(f"{save_prefix}.png", dpi=300, bbox_inches='tight')

    print(f"✅ תרשים SHAP עם טקסט מוגדל נוצר בהצלחה!")
    print(f"נתיב: {save_prefix}.png")

    plt.close(fig)

input_csv = "/mnt/new_groups/vaksler_group/Efrat/Results/experiment_runs_featuers_val_set/run_epochs_80.0_lr_0.001_din_128.0_dout_256.0_bs_64.0/Results/gems_shap_importance_models_xgbs_combine_embedding.csv"

plot_all_folds_shap_for_epoch(
    shap_csv_path=input_csv,
    target_epoch=75,
    top_n=10,
    output_dir=final_results_dir
)

################################################################################################


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_embedding_learning_dynamics(shap_csv_path, top_n=20, best_epoch=None, output_dir=None):
    """
    מייצרת גרף המראה את אחוז מאפייני ה-Embedding בתוך ה-Top N
    פיצ'רים לאורך האפוקים.
    מותאם קונספטואלית למאמר: Embeddings מיוצגים על ידי הצבע הסגול (MTI_EDGE).
    """
    if not os.path.exists(shap_csv_path):
        print(f"❌ שגיאה: הקובץ לא נמצא בנתיב: {shap_csv_path}")
        return

    print("⏳ מעבד את נתוני ה-SHAP ומחשב אחוזים, זה עשוי לקחת כמה שניות...")

    # 1. קריאת הנתונים וזיהוי העמודה הנכונה
    df = pd.read_csv(shap_csv_path)
    shap_col = [col for col in df.columns if col.startswith('mean_abs')]
    if not shap_col:
        print("❌ שגיאה: לא נמצאה עמודת SHAP.")
        return
    shap_col_name = shap_col[0]

    # 2. סידור סוגי נתונים
    df['epoch'] = df['epoch'].astype(int)
    df['fold'] = df['fold'].astype(str)

    # 3. חישוב האחוזים לכל פולד ואפוק
    results = []
    for (fold, epoch), group in df.groupby(['fold', 'epoch']):
        top_features = group.sort_values(by=shap_col_name, ascending=False).head(top_n)
        emb_count = sum(top_features['feature'].astype(str).str.startswith('dim_'))
        emb_percent = (emb_count / top_n) * 100

        results.append({
            'fold': fold,
            'epoch': epoch,
            'embedding_percentage': emb_percent
        })

    res_df = pd.DataFrame(results)

    # ==========================================
    # 4. ציור הגרף (עיצוב ברמת פרסום - Style Aligned)
    # ==========================================
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']

    fig, ax = plt.subplots(figsize=(10, 6))

    # --- Official Brand Colors ---
    MTI_EDGE = '#9966CC'       # סגול עמוק עבור ה-Embeddings (הקו המרכזי)
    MTI_LILAC = '#C8A2C8'      # סגול לילך בהיר עבור הפולדים
    REFERENCE_RED = '#E53935'  # קו הייחוס של האפוק המנצח
    BASELINE_GRAY = '#777777'

    # ציור 10 העקומות של הפולדים (סגול בהיר ושקוף למניעת עומס, ושמירה על קשר ל-Embedding)
    for fold in res_df['fold'].unique():
        fold_data = res_df[res_df['fold'] == fold]
        ax.plot(fold_data['epoch'], fold_data['embedding_percentage'],
                color=BASELINE_GRAY, alpha=0.3, lw=1.5, zorder=1) # השארתי אפור כדי לשמור על ניקיון, אפשר להחליף ל-MTI_LILAC

    # חישוב וציור המגמה הממוצעת (סגול מותג עבה, כי אנחנו מדברים על Embeddings!)
    mean_df = res_df.groupby('epoch')['embedding_percentage'].mean().reset_index()
    ax.plot(mean_df['epoch'], mean_df['embedding_percentage'],
            color=MTI_EDGE, lw=3.5, label='Mean % of Embedding Features (10 folds)', zorder=5)

    # סימון האפוק המנצח (אדום ייחוס)
    # if best_epoch:
    #     ax.axvline(x=best_epoch, color=REFERENCE_RED, linestyle='--', lw=2.5,
    #                label=f'Best Epoch', zorder=4)

    # עיצוב טקסטים וצירים
    ax.set_xlabel('Training Epoch', fontsize=14, labelpad=10)
    ax.set_ylabel(fr'% of Embedding Features in $\mathit{{Top-{top_n}}}$', fontsize=14, labelpad=10)

    # הגבלת צירים
    ax.set_ylim(0, 100)
    ax.set_xlim(res_df['epoch'].min(), res_df['epoch'].max())

    # עיצוב רשת ומסגרת
    ax.grid(True, linestyle='-', color='#EEEEEE', alpha=1.0, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.spines['left'].set_color('#DDDDDD')

    ax.tick_params(axis='both', which='major', labelsize=12)

    # מקרא נקי
    ax.legend(fontsize=12, loc='lower right', frameon=True, facecolor='white', edgecolor='#DDDDDD')

    plt.tight_layout()

    # 5. שמירה
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = '.'

    save_prefix = os.path.join(output_dir, 'GEMS_Embedding_Importance_Dynamics_OfficialColors')
    plt.savefig(f"{save_prefix}.png", dpi=300, bbox_inches='tight')

    print(f"✅ גרף הדינמיקה המעוצב נוצר ונשמר בהצלחה בנתיב:")
    print(f"   {save_prefix}.png")

    plt.close(fig)

input_csv = "/mnt/new_groups/vaksler_group/Efrat/Results/experiment_runs_featuers_val_set/run_epochs_80.0_lr_0.001_din_128.0_dout_256.0_bs_64.0/Results/gems_shap_importance_models_xgbs_combine_embedding.csv"
plot_embedding_learning_dynamics(
    shap_csv_path=input_csv,
    top_n=20,
    best_epoch=target_epoch,
    output_dir=final_results_dir
)

import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_embedding_learning_dynamics(shap_csv_path, performance_csv_path, top_n=20, best_epoch=None, output_dir=None):
    """
    מייצרת גרף המראה את אחוז מאפייני ה-Embedding בתוך ה-Top N לאורך האפוקים,
    עם ציר Y משני המציג את ביצועי המודל במקביל (Val AUC / AUC).
    גרסה חכמה: מתמודדת עם שמות עמודות משתנים (Capitalization, auc/val_auc).
    """
    if not os.path.exists(shap_csv_path):
        print(f"❌ שגיאה: קובץ SHAP לא נמצא בנתיב: {shap_csv_path}")
        return
    if not os.path.exists(performance_csv_path):
        print(f"❌ שגיאה: קובץ ביצועים לא נמצא בנתיב: {performance_csv_path}")
        return

    print("⏳ מעבד את נתוני ה-SHAP והביצועים, ומכין את הגרף הכפול...")

    # ==========================================
    # 1. עיבוד נתוני SHAP
    # ==========================================
    df_shap = pd.read_csv(shap_csv_path)
    shap_col = [col for col in df_shap.columns if col.startswith('mean_abs') or col.lower().startswith('mean_abs')]
    if not shap_col:
        print("❌ שגיאה: לא נמצאה עמודת SHAP.")
        return
    shap_col_name = shap_col[0]

    # זיהוי עמודת אפוק ב-SHAP (מתעלם מאותיות גדולות/קטנות)
    shap_epoch_col = next((col for col in df_shap.columns if col.lower() == 'epoch'), None)
    shap_fold_col = next((col for col in df_shap.columns if col.lower() == 'fold'), None)

    df_shap[shap_epoch_col] = df_shap[shap_epoch_col].astype(int)
    df_shap[shap_fold_col] = df_shap[shap_fold_col].astype(str)

    results = []
    for (fold, epoch), group in df_shap.groupby([shap_fold_col, shap_epoch_col]):
        top_features = group.sort_values(by=shap_col_name, ascending=False).head(top_n)
        emb_count = sum(top_features['feature'].astype(str).str.startswith('dim_'))
        emb_percent = (emb_count / top_n) * 100

        results.append({
            'fold': fold,
            'epoch': epoch,
            'embedding_percentage': emb_percent
        })

    res_df = pd.DataFrame(results)
    mean_shap_df = res_df.groupby('epoch')['embedding_percentage'].mean().reset_index()

    # ==========================================
    # 2. עיבוד נתוני הביצועים (Metrics) - גרסה חסינה
    # ==========================================
    df_perf = pd.read_csv(performance_csv_path)

    # הפיכת כל שמות העמודות בקובץ לאותיות קטנות כדי למנוע בעיות של רגישות לאותיות גדולות (Epoch vs epoch)
    original_cols = list(df_perf.columns)
    df_perf.columns = df_perf.columns.str.lower().str.strip()

    epoch_col = 'epoch' if 'epoch' in df_perf.columns else None

    # חיפוש חכם לעמודת ה-AUC
    auc_col = None
    if 'val_auc' in df_perf.columns:
        auc_col = 'val_auc'
    elif 'auc' in df_perf.columns:
        auc_col = 'auc'
    elif 'test_auc' in df_perf.columns:
        auc_col = 'test_auc'

    if not epoch_col or not auc_col:
        print(f"❌ שגיאה: לא הצלחתי לזהות את עמודות ה-Epoch או ה-AUC בקובץ הביצועים.")
        print(f"העמודות שנמצאו בקובץ שלך הן: {original_cols}")
        print("אנא ודאי שיש עמודה שמייצגת אפוק ועמודה שמייצגת את ה-AUC.")
        return

    # חישוב ממוצע AUC לכל אפוק
    mean_perf_df = df_perf.groupby(epoch_col)[auc_col].mean().reset_index()

    # ==========================================
    # 3. ציור הגרף הכפול
    # ==========================================
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # --- פלטת צבעים ---
    MTI_EDGE = '#9966CC'
    BASELINE_GRAY = '#777777'
    PERF_TEAL = '#008080'
    REFERENCE_RED = '#E53935'

    # --- ציר Y ראשון (שמאלי) - אחוזי SHAP ---
    for fold in res_df['fold'].unique():
        fold_data = res_df[res_df['fold'] == fold]
        ax1.plot(fold_data['epoch'], fold_data['embedding_percentage'],
                 color=BASELINE_GRAY, alpha=0.2, lw=1.5, zorder=1)

    ax1.plot(mean_shap_df['epoch'], mean_shap_df['embedding_percentage'],
             color=MTI_EDGE, lw=3.5, label='Mean % Embedding Features', zorder=5)

    ax1.set_xlabel('Training Epoch', fontsize=14, labelpad=10)
    ax1.set_ylabel(fr'% of Embedding Features in $\mathit{{Top-{top_n}}}$', fontsize=14, labelpad=10, color=MTI_EDGE)
    ax1.set_ylim(0, 100)
    ax1.set_xlim(res_df['epoch'].min(), res_df['epoch'].max())

    ax1.tick_params(axis='y', labelsize=12, labelcolor=MTI_EDGE)
    ax1.tick_params(axis='x', labelsize=12)

    ax1.grid(True, linestyle='-', color='#EEEEEE', alpha=1.0, zorder=0)
    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_color('#DDDDDD')
    ax1.spines['left'].set_color(MTI_EDGE)

    # --- ציר Y משני (ימני) - Model Performance ---
    ax2 = ax1.twinx()

    # שימוש בשם העמודה החכם שמצאנו (auc_col)
    ax2.plot(mean_perf_df[epoch_col], mean_perf_df[auc_col],
             color=PERF_TEAL, lw=3.0, linestyle='--', label=f'Mean {auc_col.upper()}', zorder=6)

    ax2.set_ylabel(f'Model Performance ({auc_col.upper()})', fontsize=14, labelpad=10, color=PERF_TEAL)
    ax2.tick_params(axis='y', labelcolor=PERF_TEAL, labelsize=12)

    ax2.spines['right'].set_color(PERF_TEAL)
    ax2.spines['top'].set_visible(False)

    min_auc = max(0.5, mean_perf_df[auc_col].min() - 0.05)
    max_auc = min(1.0, mean_perf_df[auc_col].max() + 0.02)
    ax2.set_ylim(min_auc, max_auc)

    # איחוד מקראות
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()

    if best_epoch:
        line_best = ax1.axvline(x=best_epoch, color=REFERENCE_RED, linestyle=':', lw=2.0, zorder=4,
                                label=f'Best Epoch ({best_epoch})')
        lines_1.append(line_best)
        labels_1.append(f'Best Epoch ({best_epoch})')

    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower right',
               fontsize=12, frameon=True, facecolor='white', edgecolor='#DDDDDD')

    plt.title('Feature Embedding Dependency vs. Model Performance', fontsize=16, fontweight='bold', pad=15)
    plt.tight_layout()

    # ==========================================
    # 4. שמירה
    # ==========================================
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = '.'

    save_prefix = os.path.join(output_dir, 'GEMS_Embedding_Dynamics_DualAxis')
    plt.savefig(f"{save_prefix}.png", dpi=300, bbox_inches='tight')

    print(f"✅ גרף הדינמיקה הכפול נוצר ונשמר בהצלחה בנתיב:")
    print(f"   {save_prefix}.png")
    plt.close(fig)
# ==========================================
# קריאת ההפעלה (להוסיף בתחתית הסקריפט):
# ==========================================
base_path = "/mnt/new_groups/vaksler_group/Efrat/Results/experiment_runs_featuers_val_set/run_epochs_80.0_lr_0.001_din_128.0_dout_256.0_bs_64.0/Results"
shap_csv = os.path.join(base_path, "gems_shap_importance_models_xgbs_combine_embedding.csv")
metrics_csv = os.path.join(base_path, "gems_metrics_models_xgbs_combine_embedding.csv")
final_results_dir = "/home/efrco/PHD/Goal_One/Final_Results/PartA"

# אפשר להשתמש במשתנה target_epoch שהגדרת קודם (לדוגמה 61 או 75)
plot_embedding_learning_dynamics(
    shap_csv_path=shap_csv,
    performance_csv_path=metrics_csv, # <-- הפרמטר החדש!
    top_n=20,
    best_epoch=target_epoch,
    output_dir=final_results_dir
)
# ################################################################################################################
# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from matplotlib.patches import Patch
# from matplotlib.ticker import MaxNLocator
#
#
# def plot_pooled_shap_for_epoch(shap_csv_path, target_epoch, top_n=20, output_dir=None):
#     """
#     מייצרת תרשים SHAP מסכם (Pooled) הממוצע על פני כל הפולדים.
#     מציג גרף אחד, נקי, קריא ומרשים - מושלם למאמר (Publication Quality).
#     """
#     if not os.path.exists(shap_csv_path):
#         print(f"❌ שגיאה: הקובץ לא נמצא בנתיב: {shap_csv_path}")
#         return
#
#     # 1. קריאת הנתונים
#     df = pd.read_csv(shap_csv_path)
#     shap_col = [col for col in df.columns if col.startswith('mean_abs')]
#     if not shap_col:
#         print("❌ שגיאה: לא נמצאה עמודת SHAP.")
#         return
#     shap_col_name = shap_col[0]
#
#     df['epoch'] = df['epoch'].astype(str)
#     target_epoch_str = str(target_epoch)
#
#     # סינון לאפוק הספציפי
#     epoch_data = df[df['epoch'] == target_epoch_str].copy()
#
#     if epoch_data.empty:
#         print(f"❌ שגיאה: לא נמצאו נתונים עבור אפוק {target_epoch}.")
#         return
#
#     # ==========================================
#     # 2. שלב הפולינג (Pooling): חישוב ממוצע של כל פיצ'ר על פני כל הפולדים
#     # ==========================================
#     pooled_shap = epoch_data.groupby('feature')[shap_col_name].mean().reset_index()
#
#     # מיון ובחירת ה-Top N
#     top_features = pooled_shap.sort_values(by=shap_col_name, ascending=False).head(top_n).copy()
#
#     # שמירת השם המלא
#     top_features['display_name'] = top_features['feature']
#
#     # --- Official Brand Colors ---
#     GEM_GREEN_LIGHT = '#C1E1C1'
#     GEM_GREEN_DARK = '#2E7D32'
#     MTI_LILAC = '#C8A2C8'
#     MTI_EDGE = '#9966CC'
#
#     face_colors = [MTI_LILAC if str(feat).startswith('dim_') else GEM_GREEN_LIGHT for feat in top_features['feature']]
#     edge_colors = [MTI_EDGE if str(feat).startswith('dim_') else GEM_GREEN_DARK for feat in top_features['feature']]
#
#     # ==========================================
#     # 3. ציור הגרף (פאנל אחד גדול ומרשים)
#     # ==========================================
#     plt.rcParams['font.family'] = 'sans-serif'
#     plt.rcParams['font.sans-serif'] = ['Arial']
#
#     # קנבס פרופורציונלי (רוחב יפה לברים, גובה שמכיל 20 פיצ'רים ברווח)
#     fig, ax = plt.subplots(figsize=(18, 14))
#
#     sns.barplot(
#         data=top_features,
#         x=shap_col_name,
#         y='display_name',
#         palette=face_colors,
#         ax=ax,
#         linewidth=3.0,
#         zorder=3
#     )
#
#     for i, patch in enumerate(ax.patches):
#         if i < len(edge_colors):
#             patch.set_edgecolor(edge_colors[i])
#
#     # כותרות ועיצוב אקדמי נקי
#     ax.set_title(fr'$\mathit{{SHAP}}$ Feature Importance (Pooled Across All 10 Folds, Epoch {target_epoch})',
#                  fontsize=30, fontweight='bold', color='#222222', pad=25)
#
#     ax.set_xlabel(r'Mean |$\mathit{SHAP}$| value (average impact on model output magnitude)',
#                   fontsize=20, labelpad=15)
#     ax.set_ylabel('')
#
#     # סידור ציר ה-X עם רווחים ברורים
#     ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
#     ax.tick_params(axis='x', labelsize=18)
#
#     ax.grid(axis='x', linestyle='-', color='#EEEEEE', alpha=1.0, zorder=0)
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['bottom'].set_color('#DDDDDD')
#     ax.spines['left'].set_color('#DDDDDD')
#
#     # =========================================================
#     # הטקסט מיושר שמאלה ויושב באופן מושלם ב"פדינג" שהשארנו
#     # =========================================================
#     ax.set_yticks(range(len(top_features)))
#     ax.set_yticklabels([])
#
#     # עוגן יחסי מדויק (x=-0.2) שמכניס את הטקסט לשוליים הריקים מבלי לגעת בגרף
#     for i, name in enumerate(top_features['display_name']):
#         ax.text(-0.04, i, name,
#                 ha='right', va='center',  # עבור פאנל בודד, יישור לימין ביחס לנקודה הקרובה לציר יוצר טקסט נקי וקריא
#                 fontsize=30, color='#222222',
#                 transform=ax.get_yaxis_transform())
#
#     # מקרא
#     legend_elements = [
#         Patch(facecolor=GEM_GREEN_LIGHT, edgecolor=GEM_GREEN_DARK, linewidth=3.0, label='Biological Features'),
#         Patch(facecolor=MTI_LILAC, edgecolor=MTI_EDGE, linewidth=3.0, label='Graph Embeddings Features (dim_X)')
#     ]
#
#     # במקום למעלה, בגרף יחיד לפעמים יותר יפה למקם את המקרא בצד הימני התחתון (היכן שיש חלל ריק)
#     ax.legend(handles=legend_elements, loc='lower right',
#               fontsize=25, frameon=True, facecolor='white', edgecolor='#DDDDDD', borderpad=1.0)
#
#     # יצירת שוליים שמאליים רחבים מאוד כדי להכיל את שמות הפיצ'רים הארוכים ביותר בנחת
#     plt.subplots_adjust(left=0.35, right=0.95, top=0.92, bottom=0.1)
#
#     if output_dir:
#         os.makedirs(output_dir, exist_ok=True)
#     else:
#         output_dir = '.'
#
#     save_prefix = os.path.join(output_dir, f'GEMS_SHAP_Pooled_Epoch_{target_epoch}')
#
#     plt.savefig(f"{save_prefix}.png", dpi=300, bbox_inches='tight')

#
#     print(f"✅ תרשים SHAP מסכם (Pooled) נוצר בהצלחה!")
#     print(f"נתיב: {save_prefix}.png")
#
#     plt.close(fig)
#
# # ================================
# # הפעלה לדוגמה:
# # (שימי לב שברירת המחדל היא 20 פיצ'רים במקום 10, כי בגרף בודד אפשר להראות יותר מידע)
# # ================================
# shap_file_path = input_csv = "/mnt/new_groups/vaksler_group/Efrat/Results/experiment_runs_featuers_val_set/run_epochs_80.0_lr_0.001_din_128.0_dout_256.0_bs_64.0/Results/gems_shap_importance_models_xgbs_combine_embedding.csv"
#
# final_output_folder = "/home/efrco/PHD/Goal_One/Final_Results/PartA/"
# plot_pooled_shap_for_epoch(shap_csv_path=shap_file_path, target_epoch=68, top_n=20, output_dir=final_output_folder)
#
# import os
# import pandas as pd
#
#
# def find_best_epoch_for_embeddings(shap_csv_path, top_n=10):
#     """
#     סורקת את כל האפוקים בקובץ ה-SHAP, מבצעת Pooling (ממוצע על כל הפולדים),
#     ובודקת באיזה אפוק אחוז הפיצ'רים מסוג 'dim_' ב-Top N הוא הגבוה ביותר.
#     """
#     if not os.path.exists(shap_csv_path):
#         print(f"❌ שגיאה: הקובץ לא נמצא בנתיב: {shap_csv_path}")
#         return None
#
#     # 1. קריאת הנתונים וזיהוי עמודת ה-SHAP
#     df = pd.read_csv(shap_csv_path)
#     shap_cols = [col for col in df.columns if col.startswith('mean_abs')]
#     if not shap_cols:
#         print("❌ שגיאה: לא נמצאה עמודת SHAP (מתחילה ב-'mean_abs').")
#         return None
#     shap_col_name = shap_cols[0]
#
#     # רשימה לאגירת התוצאות של כל אפוק
#     epoch_results = []
#
#     # 2. מעבר על כל אפוק בנפרד
#     epochs = df['epoch'].unique()
#     for epoch in epochs:
#         epoch_data = df[df['epoch'] == epoch]
#
#         # שלב הפולינג: ממוצע של כל פיצ'ר על פני הפולדים באפוק הנוכחי
#         pooled_shap = epoch_data.groupby('feature')[shap_col_name].mean().reset_index()
#
#         # חילוץ N הפיצ'רים המובילים
#         top_features = pooled_shap.sort_values(by=shap_col_name, ascending=False).head(top_n)
#
#         # ספירת כמה מתוכם הם dim_
#         dim_count = top_features['feature'].astype(str).str.startswith('dim_').sum()
#         dim_percentage = (dim_count / top_n) * 100.0
#
#         epoch_results.append({
#             'Epoch': epoch,
#             'Dim_Count': dim_count,
#             'Dim_Percentage': dim_percentage,
#             'Top_Features': list(top_features['feature'])
#         })
#
#     # 3. יצירת טבלת תוצאות ומיון
#     results_df = pd.DataFrame(epoch_results)
#     # נמיין לפי האחוז הגבוה ביותר, ואם יש תיקו נמיין לפי מספר האפוק (מהמוקדם למאוחר)
#     results_df['Epoch_Num'] = pd.to_numeric(results_df['Epoch'], errors='coerce')
#     results_df = results_df.sort_values(by=['Dim_Percentage', 'Epoch_Num'], ascending=[False, True]).drop(
#         columns=['Epoch_Num'])
#
#     # 4. הצגת התוצאה המנצחת
#     if not results_df.empty:
#         best_epoch = results_df.iloc[0]
#         print("\n" + "=" * 50)
#         print(f"🏆 האפוק המנצח: Epoch {best_epoch['Epoch']}")
#         print(
#             f"📊 אחוז מאפייני Graph Embeddings בטופ {top_n}: {best_epoch['Dim_Percentage']:.1f}% ({best_epoch['Dim_Count']}/{top_n})")
#         print("=" * 50 + "\n")
#
#         print("🔝 5 האפוקים המובילים ביותר:")
#         display_df = results_df[['Epoch', 'Dim_Count', 'Dim_Percentage']].head(5)
#         print(display_df.to_string(index=False))
#
#         print("\n💡 רשימת הפיצ'רים באפוק המנצח (מסודר מהחזק לחלש):")
#         for i, feat in enumerate(best_epoch['Top_Features'], 1):
#             icon = "🟣" if str(feat).startswith('dim_') else "🟢"
#             print(f"  {i}. {icon} {feat}")
#
#     return results_df
#
# # ================================
# # הפעלה לדוגמה:
# # ================================
#
# def find_best_epoch_for_embeddings(shap_csv_path, top_n=10):
#     """
#     סורקת את כל האפוקים בקובץ ה-SHAP, מבצעת Pooling (ממוצע על כל הפולדים),
#     ובודקת באיזה אפוק אחוז הפיצ'רים מסוג 'dim_' ב-Top N הוא הגבוה ביותר.
#     """
#     if not os.path.exists(shap_csv_path):
#         print(f"❌ שגיאה: הקובץ לא נמצא בנתיב: {shap_csv_path}")
#         return None
#
#     # 1. קריאת הנתונים וזיהוי עמודת ה-SHAP
#     df = pd.read_csv(shap_csv_path)
#     shap_cols = [col for col in df.columns if col.startswith('mean_abs')]
#     if not shap_cols:
#         print("❌ שגיאה: לא נמצאה עמודת SHAP (מתחילה ב-'mean_abs').")
#         return None
#     shap_col_name = shap_cols[0]
#
#     # רשימה לאגירת התוצאות של כל אפוק
#     epoch_results = []
#
#     # 2. מעבר על כל אפוק בנפרד
#     epochs = df['epoch'].unique()
#     for epoch in epochs:
#         epoch_data = df[df['epoch'] == epoch]
#
#         # שלב הפולינג: ממוצע של כל פיצ'ר על פני הפולדים באפוק הנוכחי
#         pooled_shap = epoch_data.groupby('feature')[shap_col_name].mean().reset_index()
#
#         # חילוץ N הפיצ'רים המובילים
#         top_features = pooled_shap.sort_values(by=shap_col_name, ascending=False).head(top_n)
#
#         # ספירת כמה מתוכם הם dim_
#         dim_count = top_features['feature'].astype(str).str.startswith('dim_').sum()
#         dim_percentage = (dim_count / top_n) * 100.0
#
#         epoch_results.append({
#             'Epoch': epoch,
#             'Dim_Count': dim_count,
#             'Dim_Percentage': dim_percentage,
#             'Top_Features': list(top_features['feature'])
#         })
#
#     # 3. יצירת טבלת תוצאות ומיון
#     results_df = pd.DataFrame(epoch_results)
#     # נמיין לפי האחוז הגבוה ביותר, ואם יש תיקו נמיין לפי מספר האפוק (מהמוקדם למאוחר)
#     results_df['Epoch_Num'] = pd.to_numeric(results_df['Epoch'], errors='coerce')
#     results_df = results_df.sort_values(by=['Dim_Percentage', 'Epoch_Num'], ascending=[False, True]).drop(columns=['Epoch_Num'])
#
#     # 4. הצגת התוצאה המנצחת
#     if not results_df.empty:
#         best_epoch = results_df.iloc[0]
#         print("\n" + "="*50)
#         print(f"🏆 האפוק המנצח: Epoch {best_epoch['Epoch']}")
#         print(f"📊 אחוז מאפייני Graph Embeddings בטופ {top_n}: {best_epoch['Dim_Percentage']:.1f}% ({best_epoch['Dim_Count']}/{top_n})")
#         print("="*50 + "\n")
#
#         print("🔝 5 האפוקים המובילים ביותר:")
#         display_df = results_df[['Epoch', 'Dim_Count', 'Dim_Percentage']].head(5)
#         print(display_df.to_string(index=False))
#
#         print("\n💡 רשימת הפיצ'רים באפוק המנצח (מסודר מהחזק לחלש):")
#         for i, feat in enumerate(best_epoch['Top_Features'], 1):
#             icon = "🟣" if str(feat).startswith('dim_') else "🟢"
#             print(f"  {i}. {icon} {feat}")
#
#     return results_df
#
#
# shap_file_path = input_csv = "/mnt/new_groups/vaksler_group/Efrat/Results/experiment_runs_featuers_val_set/run_epochs_80.0_lr_0.001_din_128.0_dout_256.0_bs_64.0/Results/gems_shap_importance_models_xgbs_combine_embedding.csv"
#
# final_output_folder = "/home/efrco/PHD/Goal_One/Final_Results/PartA/"
# results_table = find_best_epoch_for_embeddings(shap_csv_path=shap_file_path, top_n=30)


