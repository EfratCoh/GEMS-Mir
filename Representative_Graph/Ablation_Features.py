from matplotlib.lines import Line2D
from scipy import stats
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import os
from pathlib import Path

CSV_FILES_DIR = "/mnt/new_groups/vaksler_group/Efrat/Results/experiment_runs_featuers_val_set/run_epochs_80.0_lr_0.001_din_128.0_dout_256.0_bs_64.0/Results"

FINAL_OUTPUT_FOLDER = "/home/efrco/PHD/Goal_One/Final_Results/PartB"
TARGET_EPOCH = 61


def plot_synergy_gap_from_files(data_dir, output_dir, selected_epoch=76):
    """
    קורא את 3 קבצי התוצאות, מחשב ממוצע AUC של 10 הפולדים לכל אפוק בנפרד (Mean, לא Pooled),
    ומייצר את גרף ה-Synergy Gap.
    """
    print(f"📊 קורא נתונים מהתיקייה ומחשב ממוצעי פולדים...")
    data_path = Path(data_dir)

    files = {
        'Combined': "gems_metrics_models_xgbs_combine_embedding.csv",
        'Baseline': "gems_metrics_models_xgbs_without_embedding.csv",
        'Embedding': "gems_metrics_models_xgbs_only_embedding.csv"
    }

    processed_data = {}

    # 1. קריאת הקבצים וחישוב הממוצע (Mean) לכל אפוק על פני 10 הפולדים
    for key, filename in files.items():
        file_path = data_path / filename
        if not file_path.exists():
            print(f"❌ שגיאה: הקובץ {filename} לא נמצא בתיקייה {data_dir}")
            return

        try:
            df_raw = pd.read_csv(file_path)
            # חישוב ממוצע פשוט של מדד ה-AUC לכל אפוק (ללא פולינג)
            mean_per_epoch = df_raw.groupby('epoch')['auc'].mean().reset_index()
            # קוראים לעמודה בשם המדויק
            mean_per_epoch = mean_per_epoch.rename(columns={'auc': f'mean_auc_{key.lower()}'})
            processed_data[key] = mean_per_epoch
        except Exception as e:
            print(f"❌ שגיאה בעיבוד הקובץ {filename}: {e}")
            return

    # 2. איחוד הנתונים לטבלה אחת
    df = processed_data['Combined'].merge(processed_data['Baseline'], on='epoch').merge(processed_data['Embedding'],
                                                                                        on='epoch')

    # שליפת הסדרות לצורך ציור הגרפים
    combined_series = df['mean_auc_combined']
    emb_series = df['mean_auc_embedding']
    baseline_series = df['mean_auc_baseline']

    # רפרנס עבור ציר הדלתא (ממוצע הבייסליין)
    baseline_ref = baseline_series.mean()

    # הגדרות צבעים (Maroon למשולב, Turquoise לאימבדינג)
    STYLE_COLORS = {
        "Baseline": "#6c757d",  # אפור כהה
        "Combined": "#800000",  # Maroon (בורדו)
        "Embedding-only": "#40E0D0"  # Turquoise (טורקיז)
    }

    # ==========================
    # ציור הגרף
    # ==========================
    fig, ax1 = plt.subplots(figsize=(10, 6))

    l1, = ax1.plot(df['epoch'], baseline_series, color=STYLE_COLORS["Baseline"],
                   linestyle='-', linewidth=1.5, label='Handcrafted Features (Baseline)')

    l2, = ax1.plot(df['epoch'], combined_series, color=STYLE_COLORS["Combined"],
                   linewidth=2.5, label='GEMS-Mir')

    l3, = ax1.plot(df['epoch'], emb_series, color=STYLE_COLORS["Embedding-only"],
                   linewidth=2.5, linestyle='--', label='GCN Embedding Only')

    ax1.fill_between(df['epoch'], baseline_series, combined_series,
                     where=(combined_series >= baseline_series),
                     color=STYLE_COLORS["Combined"], alpha=0.1)

    ax1.fill_between(df['epoch'], baseline_series, emb_series,
                     color=STYLE_COLORS["Embedding-only"], alpha=0.05)

    ax1.set_xlabel('Training Epochs', fontweight='bold')
    ax1.set_ylabel('Mean Test AUC', color='#333333', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#333333')
    ax1.grid(True, linestyle=':', alpha=0.5)

    # ==========================
    # ציר ימין: דלתא
    # ==========================
    ax2 = ax1.twinx()

    y_min_data = min(emb_series.min(), baseline_series.min())
    y_max_data = max(combined_series.max(), baseline_series.max())
    padding = (y_max_data - y_min_data) * 0.20

    ylim_bottom = y_min_data - padding
    ylim_top = y_max_data + padding

    ax1.set_ylim(ylim_bottom, ylim_top)
    ax2.set_ylim(ylim_bottom - baseline_ref, ylim_top - baseline_ref)

    ax2.set_ylabel('Δ AUC (Relative to Features Baseline)', color='#555555', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#555555')

    ax2.axhline(0, color='gray', linewidth=0.8, alpha=0.4, zorder=0)

    # ==========================
    # סימונים (Annotations)
    # ==========================
    if selected_epoch in df['epoch'].values:
        val_comb = combined_series[df['epoch'] == selected_epoch].values[0]
        val_baseline = baseline_series[df['epoch'] == selected_epoch].values[0]

        ax1.axvline(x=selected_epoch, color='gray', linestyle=':', alpha=0.5)

        ax1.scatter(selected_epoch, val_comb, color=STYLE_COLORS["Combined"], s=90, zorder=10, edgecolors='white',
                    linewidth=1.5)
        ax1.scatter(selected_epoch, val_baseline, color=STYLE_COLORS["Baseline"], s=90, zorder=10, edgecolors='white',
                    linewidth=1.5)

        ax1.annotate('',
                     xy=(selected_epoch, val_comb),
                     xytext=(selected_epoch, val_baseline),
                     arrowprops=dict(arrowstyle='<|-|>', color='#444444', lw=1.5, mutation_scale=15))

        mid_y = (val_comb + val_baseline) / 2
        ax1.text(selected_epoch - 2, mid_y, 'Synergy\nGain',
                 horizontalalignment='right',
                 verticalalignment='center',
                 color='#444444', fontweight='bold', fontsize=11,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8))

    # ==========================
    # סיום ושמירה
    # ==========================
    legend_elements = [l1, l3, l2]
    ax1.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 1.02),
               ncol=3, frameon=False, fontsize=11)

    sns.despine(ax=ax1, top=True, right=False)
    sns.despine(ax=ax2, top=True, left=True, right=False)

    plt.title('Performance Synergy: GCN Embeddings & Handcrafted Features', fontweight='bold', y=1.12)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_file_png = os.path.join(output_dir, f'GEMS_Synergy_Gap_Epoch_{selected_epoch}.png')

    plt.savefig(out_file_png, dpi=300, bbox_inches='tight')

    print(f"✅ הגרף נשמר בהצלחה בנתיב:\n{out_file_png}")
    plt.show()
    plt.close()

# plot_synergy_gap_from_files(CSV_FILES_DIR, FINAL_OUTPUT_FOLDER, TARGET_EPOCH)



def run_wilcoxon_and_extract_all_metrics(csv_dir, target_epoch=68):
    print(f"📊 שולף את כל המדדים ומריץ מבחן Wilcoxon לאפוק {target_epoch}...\n")
    data_path = Path(csv_dir)

    files = {
        'Baseline': "gems_metrics_models_xgbs_without_embedding.csv",
        'Embedding': "gems_metrics_models_xgbs_only_embedding.csv",
        'Combined': "gems_metrics_models_xgbs_combine_embedding.csv"
    }

    data = {}

    # קריאת הנתונים וסידור לפי פולד
    for key, filename in files.items():
        file_path = data_path / filename
        if not file_path.exists():
            print(f"❌ שגיאה: הקובץ {filename} לא נמצא.")
            return

        df = pd.read_csv(file_path)
        df_epoch = df[df['epoch'] == target_epoch].sort_values('fold')

        if df_epoch.empty:
            print(f"❌ שגיאה: לא נמצאו נתונים לאפוק {target_epoch} במודל {key}.")
            return

        data[key] = df_epoch

    print("=" * 70)
    print(" 1. FULL METRICS SUMMARY (10-Fold CV) ")
    print("=" * 70)

    metrics = {}
    for key in ['Baseline', 'Embedding', 'Combined']:
        # שליפת כל המדדים
        # ⚠️ ודאי ששמות העמודות (accuracy, f1) תואמים לקובץ שלך!
        auc_vals = data[key]['auc'].values
        auprc_vals = data[key]['auprc'].values
        acc_vals = data[key]['accuracy'].values  # אם קראת לזה 'acc', שני כאן
        f1_vals = data[key]['f1_score'].values  # אם קראת לזה 'f1_score', שני כאן

        metrics[key] = {
            'auc_vals': auc_vals,
            'auc_mean': auc_vals.mean(), 'auc_std': auc_vals.std(),
            'auprc_mean': auprc_vals.mean(), 'auprc_std': auprc_vals.std(),
            'acc_mean': acc_vals.mean(), 'acc_std': acc_vals.std(),
            'f1_mean': f1_vals.mean(), 'f1_std': f1_vals.std()
        }

        name_display = "GEMS-Mir" if key == 'Combined' else "Handcrafted Features" if key == 'Baseline' else "GCN Embedding Only"
        print(f"🔸 {name_display}:")
        print(f"   AUC      = {metrics[key]['auc_mean']:.4f} ± {metrics[key]['auc_std']:.4f}")
        print(f"   AUPRC    = {metrics[key]['auprc_mean']:.4f} ± {metrics[key]['auprc_std']:.4f}")
        print(f"   Accuracy = {metrics[key]['acc_mean']:.4f} ± {metrics[key]['acc_std']:.4f}")
        print(f"   F1-Score = {metrics[key]['f1_mean']:.4f} ± {metrics[key]['f1_std']:.4f}\n")

    print("=" * 70)
    print(" 2. WILCOXON SIGNED-RANK TEST (on AUC) ")
    print("=" * 70)

    auc_comb = metrics['Combined']['auc_vals']

    # השוואה 1: Combined vs Baseline
    auc_base = metrics['Baseline']['auc_vals']
    stat_base, p_val_base = stats.wilcoxon(auc_comb, auc_base)
    sig_base = "***" if p_val_base < 0.001 else "**" if p_val_base < 0.01 else "*" if p_val_base < 0.05 else "ns"

    print(f"🟢 GEMS-MIR vs. Handcrafted Features (Baseline):")
    print(f"   Wilcoxon Statistic: {stat_base:.1f}")
    print(f"   P-value           : {p_val_base:.4e} ({sig_base})")
    print(f"   Conclusion        : {'Significant Improvement' if p_val_base < 0.05 else 'Not Significant'}\n")

    # השוואה 2: Combined vs Embedding
    auc_emb = metrics['Embedding']['auc_vals']
    stat_emb, p_val_emb = stats.wilcoxon(auc_comb, auc_emb)
    sig_emb = "***" if p_val_emb < 0.001 else "**" if p_val_emb < 0.01 else "*" if p_val_emb < 0.05 else "ns"

    print(f"🟢 GEMS-MIR vs. GCN Embedding Only:")
    print(f"   Wilcoxon Statistic: {stat_emb:.1f}")
    print(f"   P-value           : {p_val_emb:.4e} ({sig_emb})")
    print(f"   Conclusion        : {'Significant Improvement' if p_val_emb < 0.05 else 'Not Significant'}")
    print("=" * 70)


run_wilcoxon_and_extract_all_metrics(CSV_FILES_DIR, TARGET_EPOCH)


def plot_synergy_line_with_exact_values(data_dir, output_dir, selected_epoch=68):
    print(f"📊 קורא נתונים מהתיקייה ומחשב ממוצעי פולדים...")
    data_path = Path(data_dir)

    files = {
        'Combined': "gems_metrics_models_xgbs_combine_embedding.csv",
        'Baseline': "gems_metrics_models_xgbs_without_embedding.csv",
        'Embedding': "gems_metrics_models_xgbs_only_embedding.csv"
    }

    processed_data = {}

    # 1. קריאת הקבצים וחישוב הממוצע לכל אפוק
    for key, filename in files.items():
        file_path = data_path / filename
        if not file_path.exists():
            print(f"❌ שגיאה: הקובץ {filename} לא נמצא בתיקייה.")
            return

        try:
            df_raw = pd.read_csv(file_path)
            mean_per_epoch = df_raw.groupby('epoch')['auc'].mean().reset_index()
            mean_per_epoch = mean_per_epoch.rename(columns={'auc': f'mean_auc_{key.lower()}'})
            processed_data[key] = mean_per_epoch
        except Exception as e:
            print(f"❌ שגיאה בעיבוד הקובץ {filename}: {e}")
            return

    # 2. איחוד הנתונים לטבלה אחת
    df = processed_data['Combined'].merge(processed_data['Baseline'], on='epoch').merge(processed_data['Embedding'],
                                                                                        on='epoch')

    combined_series = df['mean_auc_combined']
    emb_series = df['mean_auc_embedding']
    baseline_series = df['mean_auc_baseline']
    baseline_ref = baseline_series.mean()

    # צבעים קלאסיים (אדום, כחול, אפור) כפי שהיו בפיגר שאהבת
    STYLE_COLORS = {
        "Baseline": "#6c757d",  # אפור כהה
        "Combined": "#d90429",  # אדום עמוק
        "Embedding-only": "#0077b6"  # כחול
    }

    # ==========================
    # ציור הגרף (ציר שמאל)
    # ==========================
    fig, ax1 = plt.subplots(figsize=(10.5, 6))

    l1, = ax1.plot(df['epoch'], baseline_series, color=STYLE_COLORS["Baseline"],
                   linestyle='-', linewidth=1.5, label='Handcrafted Features (Baseline)')

    l2, = ax1.plot(df['epoch'], combined_series, color=STYLE_COLORS["Combined"],
                   linewidth=2.5, label='GEMS-Mir')

    l3, = ax1.plot(df['epoch'], emb_series, color=STYLE_COLORS["Embedding-only"],
                   linewidth=2.5, linestyle='--', label='GCN Embedding Only')

    ax1.fill_between(df['epoch'], baseline_series, combined_series,
                     where=(combined_series >= baseline_series),
                     color=STYLE_COLORS["Combined"], alpha=0.1)

    ax1.fill_between(df['epoch'], baseline_series, emb_series,
                     color=STYLE_COLORS["Embedding-only"], alpha=0.05)

    ax1.set_xlabel('Training Epochs', fontweight='bold')
    ax1.set_ylabel('Mean Test AUC', color='#333333', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#333333')
    ax1.grid(True, linestyle=':', alpha=0.5)

    # הרחבת ציר ה-X קצת ימינה כדי שהטקסט של המספרים לא ייחתך
    max_epoch = df['epoch'].max()
    ax1.set_xlim(left=0, right=max_epoch + 10)

    # ==========================
    # ציור הגרף (ציר ימין - דלתא)
    # ==========================
    ax2 = ax1.twinx()

    y_min_data = min(emb_series.min(), baseline_series.min())
    y_max_data = max(combined_series.max(), baseline_series.max())
    padding = (y_max_data - y_min_data) * 0.20

    ylim_bottom = y_min_data - padding
    ylim_top = y_max_data + padding

    ax1.set_ylim(ylim_bottom, ylim_top)
    ax2.set_ylim(ylim_bottom - baseline_ref, ylim_top - baseline_ref)

    ax2.set_ylabel('Δ AUC (Relative to Features Baseline)', color='#555555', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#555555')
    ax2.axhline(0, color='gray', linewidth=0.8, alpha=0.4, zorder=0)

    # ==========================
    # סימונים מדויקים לאפוק הנבחר (מה שביקשת!)
    # ==========================
    if selected_epoch in df['epoch'].values:
        val_comb = combined_series[df['epoch'] == selected_epoch].values[0]
        val_baseline = baseline_series[df['epoch'] == selected_epoch].values[0]

        # קו אנכי באפוק הנבחר
        ax1.axvline(x=selected_epoch, color='gray', linestyle=':', alpha=0.7)

        # נקודות מודגשות
        ax1.scatter(selected_epoch, val_comb, color=STYLE_COLORS["Combined"], s=100, zorder=10, edgecolors='white',
                    linewidth=1.5)
        ax1.scatter(selected_epoch, val_baseline, color=STYLE_COLORS["Baseline"], s=100, zorder=10, edgecolors='white',
                    linewidth=1.5)

        # חץ Synergy
        ax1.annotate('', xy=(selected_epoch, val_comb), xytext=(selected_epoch, val_baseline),
                     arrowprops=dict(arrowstyle='<|-|>', color='#444444', lw=1.5, mutation_scale=15))

        # כיתוב 'Synergy Gain' משמאל לקו
        mid_y = (val_comb + val_baseline) / 2
        ax1.text(selected_epoch - 2, mid_y, 'Synergy\nGain',
                 ha='right', va='center', color='#444444', fontweight='bold', fontsize=11,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8))

        # --- התוספת החדשה: סימון הערכים המדויקים על הגרף! ---
        # טקסט למודל המשולב
        ax1.annotate(f'AUC: {val_comb:.4f}',
                     xy=(selected_epoch, val_comb), xytext=(8, 0), textcoords="offset points",
                     ha='left', va='center', fontsize=11, fontweight='bold', color=STYLE_COLORS["Combined"],
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=STYLE_COLORS["Combined"], alpha=0.9))

        # טקסט לבייסליין
        ax1.annotate(f'AUC: {val_baseline:.4f}',
                     xy=(selected_epoch, val_baseline), xytext=(8, 0), textcoords="offset points",
                     ha='left', va='center', fontsize=11, fontweight='bold', color=STYLE_COLORS["Baseline"],
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=STYLE_COLORS["Baseline"], alpha=0.9))

    # ==========================
    # Legend וסיום
    # ==========================
    legend_elements = [l1, l3, l2]
    ax1.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 1.02),
               ncol=3, frameon=False, fontsize=11)

    sns.despine(ax=ax1, top=True, right=False)
    sns.despine(ax=ax2, top=True, left=True, right=False)

    plt.title('Performance Synergy: GCN Embeddings & Handcrafted Features', fontweight='bold', y=1.12)
    plt.tight_layout()

    # שמירה
    os.makedirs(output_dir, exist_ok=True)
    out_file_png = os.path.join(output_dir, f'GEMS_Synergy_Values_Epoch_{selected_epoch}.png')

    plt.savefig(out_file_png, dpi=300, bbox_inches='tight')


    print(f"✅ הגרף נשמר בהצלחה בנתיב:\n{out_file_png}")
    plt.show()
    plt.close()


# plot_synergy_line_with_exact_values(CSV_FILES_DIR, FINAL_OUTPUT_FOLDER, TARGET_EPOCH)

# PAY ATTATION - The auc is the mean values across 10 folds. But! the curve auc are pass intrpolation of the values in
# order to create the curv. So the AUC is not the area under the green curve
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc

def plot_combined_synergy_and_roc(metrics_dir, preds_dir, output_dir, target_epoch=74):
    print(f"📊 מייצר פיגר כפול (A+B) ברמת פרסום (ללא כותרות, רק A/B) עבור אפוק {target_epoch}...")

    # --- צבעי המותג הרשמיים המעודכנים ---
    STYLE_COLORS = {
        "Baseline": "#E53935",   # אדום (Handcrafted Features Baseline)
        "Embedding": "#9966CC",  # סגול (Graph Embeddings) - נשאר ללא שינוי
        "Combined": "#4CAF50",   # ירוק (GEMS-MIR Combined) - ייצר הצללה ירקרקה
        "Grid": "#EEEEEE",
        "Text": "#333333"
    }

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']

    # יצירת הקנבס הכפול
    fig, (ax_lines, ax_curves) = plt.subplots(1, 2, figsize=(20, 8))

    # =================================================================
    # פאנל A: Synergy Line Plot (מדדים)
    # =================================================================
    metrics_files = {
        'Combined': "gems_metrics_models_xgbs_combine_embedding.csv",
        'Baseline': "gems_metrics_models_xgbs_without_embedding.csv",
        'Embedding': "gems_metrics_models_xgbs_only_embedding.csv"
    }

    df_list = []
    for key, filename in metrics_files.items():
        file_path = Path(metrics_dir) / filename
        if not file_path.exists():
            continue
        df_raw = pd.read_csv(file_path)
        mean_per_epoch = df_raw.groupby('epoch')['auc'].mean().reset_index()
        mean_per_epoch = mean_per_epoch.rename(columns={'auc': f'mean_auc_{key.lower()}'})
        df_list.append(mean_per_epoch)

    # איחוד טבלאות
    if len(df_list) == 3:
        df_metrics = df_list[0].merge(df_list[1], on='epoch').merge(df_list[2], on='epoch')

        # ציור הקווים
        l1, = ax_lines.plot(df_metrics['epoch'], df_metrics['mean_auc_baseline'], color=STYLE_COLORS["Baseline"],
                            linestyle='-', linewidth=2.5, label='Handcrafted-Only')
        l3, = ax_lines.plot(df_metrics['epoch'], df_metrics['mean_auc_embedding'], color=STYLE_COLORS["Embedding"],
                            linewidth=2.5, linestyle='--', label='Embedding-Only')
        l2, = ax_lines.plot(df_metrics['epoch'], df_metrics['mean_auc_combined'], color=STYLE_COLORS["Combined"],
                            linewidth=3.5, label='GEMS-Mir', zorder=4)

        # הוספת הצללות (כעת הצבע של Combined ייצר שקיפות ירקרקה)
        ax_lines.fill_between(df_metrics['epoch'], df_metrics['mean_auc_baseline'], df_metrics['mean_auc_combined'],
                              where=(df_metrics['mean_auc_combined'] >= df_metrics['mean_auc_baseline']),
                              color=STYLE_COLORS["Combined"], alpha=0.1)
        ax_lines.fill_between(df_metrics['epoch'], df_metrics['mean_auc_baseline'], df_metrics['mean_auc_embedding'],
                              color=STYLE_COLORS["Embedding"], alpha=0.05)

        # עיצוב צירים
        ax_lines.set_xlabel('Training Epochs', fontsize=15, labelpad=10, color=STYLE_COLORS["Text"])
        ax_lines.set_ylabel(r'Mean Test $\mathit{AUC}$', color=STYLE_COLORS["Text"], fontsize=15, labelpad=10)
        ax_lines.grid(True, linestyle='-', color=STYLE_COLORS["Grid"], alpha=1.0, zorder=0)
        ax_lines.set_xlim(left=0, right=df_metrics['epoch'].max() + 5)

        # # נעיצת הנתונים באפוק הנבחר
        # if target_epoch in df_metrics['epoch'].values:
        #     val_comb = df_metrics.loc[df_metrics['epoch'] == target_epoch, 'mean_auc_combined'].values[0]
        #     val_base = df_metrics.loc[df_metrics['epoch'] == target_epoch, 'mean_auc_baseline'].values[0]
        #
        #     ax_lines.axvline(x=target_epoch, color='#999999', linestyle=':', linewidth=2, alpha=0.8, zorder=1)
        #
        #     # הנקודות יקבלו את הצבעים החדשים אוטומטית
        #     ax_lines.scatter(target_epoch, val_comb, color=STYLE_COLORS["Combined"], s=120, zorder=10, edgecolors='white', linewidth=2)
        #     ax_lines.scatter(target_epoch, val_base, color=STYLE_COLORS["Baseline"], s=120, zorder=10, edgecolors='white', linewidth=2)
        #
        #     # חץ סינרגיה
        #     ax_lines.annotate('', xy=(target_epoch, val_comb), xytext=(target_epoch, val_base),
        #                       arrowprops=dict(arrowstyle='<|-|>', color='#555555', lw=1.5))
        #     ax_lines.text(target_epoch - 1.5, (val_comb + val_base) / 2, 'Synergy\nGain', ha='right', va='center',
        #                   fontsize=11, fontweight='bold', color='#555555',
        #                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.9))
        #
        #     # תוויות ערכים עם הרקעים המעודכנים
        #     ax_lines.annotate(f'{val_comb:.3f}', xy=(target_epoch, val_comb), xytext=(12, 0), textcoords="offset points",
        #                       fontsize=12, fontweight='bold', color='white', va='center',
        #                       bbox=dict(boxstyle="round,pad=0.3", fc=STYLE_COLORS["Combined"], ec="none"))
        #     ax_lines.annotate(f'{val_base:.3f}', xy=(target_epoch, val_base), xytext=(12, 0), textcoords="offset points",
        #                       fontsize=12, fontweight='bold', color='white', va='center',
        #                       bbox=dict(boxstyle="round,pad=0.3", fc=STYLE_COLORS["Baseline"], ec="none"))

        ax_lines.legend(handles=[l1, l3, l2], loc='lower right',
                        frameon=True, facecolor='white', edgecolor='#DDDDDD', fontsize=13, borderpad=0.8)

    # =================================================================
    # פאנל B: ROC Curves (חיזויים)
    # =================================================================
    preds_files = {
        'Baseline': "gems_predictions_models_xgbs_without_embedding.csv",
        'Embedding': "gems_predictions_models_xgbs_only_embedding.csv",
        'Combined': "gems_predictions_models_xgbs_combine_embedding.csv"
    }


    for key, filename in preds_files.items():
        file_path = Path(preds_dir) / filename
        if not file_path.exists():
            continue

        df_pred = pd.read_csv(file_path)
        df_pred_epoch = df_pred[df_pred['epoch'] == target_epoch]

        if df_pred_epoch.empty:
            continue

        y_true = df_pred_epoch['True_Label']
        y_prob = df_pred_epoch['Prob_Score']

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_score = auc(fpr, tpr)

        label_text = 'Handcrafted-only' if key == 'Baseline' else 'Embedding-Only' if key == 'Embedding' else 'GEMS-Mir'
        line_style = '--' if key == 'Embedding' else '-'
        line_width = 3.5 if key == 'Combined' else 2.5

        plot_color = STYLE_COLORS[key]
        ax_curves.plot(fpr, tpr, color=plot_color, linestyle=line_style, linewidth=line_width,
                       label=fr'{label_text} ($\mathit{{AUC}}$ = {auc_score:.3f})')

    ax_curves.plot([0, 1], [0, 1], color='#999999', linestyle='--', linewidth=2, label='Random Classifier (0.500)')

    ax_curves.set_xlabel('False Positive Rate', fontsize=15, labelpad=10, color=STYLE_COLORS["Text"])
    ax_curves.set_ylabel('True Positive Rate', fontsize=15, labelpad=10, color=STYLE_COLORS["Text"])

    ax_curves.grid(True, linestyle='-', color=STYLE_COLORS["Grid"], alpha=1.0, zorder=0)
    ax_curves.set_xlim([-0.02, 1.02])
    ax_curves.set_ylim([-0.02, 1.02])
    ax_curves.legend(loc='lower right', fontsize=13, frameon=True, facecolor='white', edgecolor='#DDDDDD', borderpad=0.8)

    # --- עיצוב אחיד לשני הפאנלים + הוספת A ו-B בצורה תקנית ---
    for ax, letter in zip([ax_lines, ax_curves], ['A', 'B']):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#DDDDDD')
        ax.spines['left'].set_color('#DDDDDD')
        ax.tick_params(axis='both', which='major', labelsize=13)

        # מיקום אותיות A, B
        ax.text(-0.08, 1.02, letter, transform=ax.transAxes,
                fontsize=24, fontweight='bold', va='top', ha='right', color=STYLE_COLORS["Text"])

    # סיום ושמירה
    plt.tight_layout(w_pad=4.0)

    os.makedirs(output_dir, exist_ok=True)
    out_png = Path(output_dir) / f'GEMS_Panels_AB_Epoch_{target_epoch}_PublicationReady.png'
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f"✅ הפיגר הכפול נשמר בנתיב:\n{out_png}")
    plt.close(fig)

# קריאה לפונקציה (יש לעדכן את הנתיבים בהתאם לתיקיות שלך)
plot_combined_synergy_and_roc(CSV_FILES_DIR, CSV_FILES_DIR, FINAL_OUTPUT_FOLDER, TARGET_EPOCH)


#####################################Delong on the no calibrated AUC ############################################################

import pandas as pd
import numpy as np
import scipy.stats
import os


# ======================================================================
# 1. אלגוריתם DeLong לחישוב p-value (ללא שינוי)
# ======================================================================
def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    return T2


def fastDeLong(predictions_sq, labels):
    m = predictions_sq.shape[1]
    positives = predictions_sq[labels == 1]
    negatives = predictions_sq[labels == 0]

    m_pos, m_neg = len(positives), len(negatives)

    tx = np.empty([m_pos, m])
    ty = np.empty([m_neg, m])
    tz = np.empty([m_pos + m_neg, m])

    for r in range(m):
        tx[:, r] = compute_midrank(positives[:, r])
        ty[:, r] = compute_midrank(negatives[:, r])
        tz[:, r] = compute_midrank(predictions_sq[:, r])

    aucs = tz[labels == 1].sum(axis=0) / (m_pos * m_neg) - (m_pos + 1) / (2.0 * m_neg)
    v01 = (tz[labels == 1] - tx) / m_neg
    v10 = 1.0 - (tz[labels == 0] - ty) / m_pos

    sx = np.cov(v01, rowvar=False)
    sy = np.cov(v10, rowvar=False)
    delongcov = sx / m_pos + sy / m_neg
    return aucs, delongcov


def calc_pvalue(aucs, cov):
    l = np.array([[1, -1]])
    variance = np.dot(np.dot(l, cov), l.T)
    if variance[0][0] == 0:
        return 1.0
    z = np.abs(np.diff(aucs)) / np.sqrt(variance)
    return 2 * (1 - scipy.stats.norm.cdf(z))[0][0]


def delong_roc_test(y_true, y_pred1, y_pred2):
    aucs, cov = fastDeLong(np.column_stack((y_pred1, y_pred2)), y_true)
    return calc_pvalue(aucs, cov), aucs


# ======================================================================
# 2. הגדרות ונתיבים
# ======================================================================
TARGET_EPOCH = 61
CSV_FILES_DIR = "/mnt/new_groups/vaksler_group/Efrat/Results/experiment_runs_featuers_val_set/run_epochs_80.0_lr_0.001_din_128.0_dout_256.0_bs_64.0/Results"

PRED_FILES = {
    "Combined": "gems_predictions_models_xgbs_combine_embedding.csv",
    "Baseline": "gems_predictions_models_xgbs_without_embedding.csv",
    "Embedding": "gems_predictions_models_xgbs_only_embedding.csv"
}

# ======================================================================
# 3. טעינת הנתונים וסינון לאפוק הנבחר
# ======================================================================
dfs = {}
for key, filename in PRED_FILES.items():
    file_path = os.path.join(CSV_FILES_DIR, filename)
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        if 'epoch' in df.columns:
            df = df[df['epoch'] == TARGET_EPOCH].copy()

        fold_col = 'Fold' if 'Fold' in df.columns else ('fold' if 'fold' in df.columns else None)
        true_col = next((c for c in df.columns if c.lower() == 'true_label'), None)
        prob_col = next((c for c in df.columns if c.lower() == 'prob_score'), None)

        df = df.rename(columns={true_col: 'True_Label', prob_col: 'Prob_Score', fold_col: 'fold_num'})
        df['fold_num'] = df['fold_num'].astype(str).str.extract(r'(\d+)').astype(int)

        dfs[key] = df[['fold_num', 'True_Label', 'Prob_Score']]
    else:
        print(f"⚠️ הקובץ לא נמצא: {file_path}")

# ======================================================================
# 4. Pooling ישר של ההסתברויות (ללא כיול)
# ======================================================================
if len(dfs) == 3:
    print(f"🔍 מתחיל תהליך Pooling גולמי (ללא כיול) עבור Epoch {TARGET_EPOCH}...\n")

    pooled_data = {"True_Label": [], "Combined": [], "Baseline": [], "Embedding": []}
    folds = sorted(dfs["Combined"]['fold_num'].unique())

    for f in folds:
        d_comb = dfs["Combined"][dfs["Combined"]['fold_num'] == f].reset_index(drop=True)
        d_base = dfs["Baseline"][dfs["Baseline"]['fold_num'] == f].reset_index(drop=True)
        d_emb = dfs["Embedding"][dfs["Embedding"]['fold_num'] == f].reset_index(drop=True)

        if len(d_comb) == len(d_base) == len(d_emb) and len(d_comb) > 0:
            y_true = d_comb['True_Label'].values
            assert np.array_equal(
                d_comb['True_Label'].values,
                d_base['True_Label'].values
            )

            assert np.array_equal(
                d_comb['True_Label'].values,
                d_emb['True_Label'].values
            )
            # הוספה ישירה של התחזיות הגולמיות
            pooled_data["True_Label"].extend(y_true)
            pooled_data["Combined"].extend(d_comb['Prob_Score'].values)
            pooled_data["Baseline"].extend(d_base['Prob_Score'].values)
            pooled_data["Embedding"].extend(d_emb['Prob_Score'].values)
        else:
            print(f"⚠️ דילוג על פולד {f} עקב חוסר התאמה במספר הדגימות.")

    # המרה למערכי Numpy
    y_true_pooled = np.array(pooled_data["True_Label"])
    y_comb_pooled = np.array(pooled_data["Combined"])
    y_base_pooled = np.array(pooled_data["Baseline"])
    y_emb_pooled = np.array(pooled_data["Embedding"])

    print(f"✅ Pooling הושלם. סך הכל דגימות מכל הפולדים: {len(y_true_pooled)}\n")

    # ======================================================================
    # 5. הרצת DeLong על הנתונים המאוחדים
    # ======================================================================
    print("=" * 60)
    print("📊 RAW POOLED DELONG TEST RESULTS (No Calibration)")
    print("=" * 60)

    # Combined vs Baseline
    pval_base, aucs_base = delong_roc_test(y_true_pooled, y_comb_pooled, y_base_pooled)
    print(f"1. GEMS-MIR (Combined) vs. Handcrafted Features (Baseline):")
    print(f"   Combined AUC: {aucs_base[0]:.4f} | Baseline AUC: {aucs_base[1]:.4f}")
    print(f"   DeLong p-value: {pval_base:.4e}")
    if pval_base < 0.05:
        print("   ✅ תוצאה מובהקת סטטיסטית!")

    print("-" * 60)

    # Combined vs Embedding
    pval_emb, aucs_emb = delong_roc_test(y_true_pooled, y_comb_pooled, y_emb_pooled)
    print(f"2. GEMS-MIR (Combined) vs. GCN Embeddings Only:")
    print(f"   Combined AUC: {aucs_emb[0]:.4f} | Embedding AUC: {aucs_emb[1]:.4f}")
    print(f"   DeLong p-value: {pval_emb:.4e}")
    if pval_emb < 0.05:
        print("   ✅ תוצאה מובהקת סטטיסטית!")
    print("=" * 60)

else:
    print("⚠️ Please ensure all three prediction files are successfully loaded.")





