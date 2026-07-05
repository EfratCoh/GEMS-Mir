import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import numpy as np
from pathlib import Path


# ==========================================
#   GLOBAL STYLE SETTINGS
# ==========================================
sns.set_style("ticks", {"axes.grid": True, "grid.linestyle": ":", "grid.alpha": 0.5})
sns.set_context("paper", font_scale=1.4)

STYLE_COLORS = {
    "Baseline": "#377eb8",  # Blue
    "Combined": "#e41a1c",  # Red
    "Embedding-only": "#4daf4a",  # Green
    "Chance": "#999999",  # Grey
    "Highlight": "#ff7f00"  # Orange (for selected epoch)
}


# ==========================================
#   Function 1: Delta AUC Distribution
# ==========================================
def plot_delta_auc_distribution(csv_dir, csv_path, selected_epoch=76):
    """
    Plots the distribution of Delta AUC for a specific epoch across all folds.
    """
    print(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("Error: The file 'delong_results.csv' was not found.")
        return

    if selected_epoch not in df['epoch'].values:
        print(f"Error: Epoch {selected_epoch} not found.")
        return

    epoch_data = df[df['epoch'] == selected_epoch].copy()
    mean_delta = epoch_data['delta_auc'].mean()

    plt.figure(figsize=(7, 6))

    # Box Plot (חצי שקוף)
    sns.boxplot(y=epoch_data['delta_auc'],
                color=STYLE_COLORS["Combined"],
                width=0.4, linewidth=1.5, fliersize=0,
                boxprops=dict(alpha=0.3))  # שקיפות לתיבה

    # Strip Plot (הנקודות)
    sns.stripplot(y=epoch_data['delta_auc'],
                  color=STYLE_COLORS["Combined"],
                  size=8, jitter=0.1, alpha=0.8)

    # הממוצע (יהלום)
    plt.scatter(x=0, y=mean_delta, color='white', marker='D', s=100,
                edgecolors='black', zorder=10, label=f'Mean ({mean_delta:.3f})')

    plt.title(f'Improvement Distribution (Epoch {selected_epoch})', fontweight='bold')
    plt.ylabel(r'$\Delta$ AUC (Combined - Baseline)')
    plt.xticks([])  # הסרת X

    # מקרא מותאם אישית
    legend_elements = [
        Line2D([0], [0], color=STYLE_COLORS["Combined"], lw=4, alpha=0.3, label='IQR'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=STYLE_COLORS["Combined"],
               markersize=8, label='Individual Fold'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='w', markeredgecolor='k',
               markersize=8, label=f'Mean ({mean_delta:.3f})')
    ]
    plt.legend(handles=legend_elements, loc='upper right', frameon=False, fontsize=11)

    sns.despine(bottom=True)  # הסרת קו תחתון כי אין ציר X
    plt.tight_layout()
    plt.savefig(csv_dir /'delta_auc_clean.png', dpi=300, bbox_inches='tight')
    print("Saved: delta_auc_clean.png")
    plt.close()


# ==========================================
#   Function 2: Learning Dynamics
# ==========================================
def plot_learning_dynamics(csv_dir, csv_path, selected_epoch=76):
    """
    Plots the learning curve (AUC over epochs) compared to a static baseline.
    """
    print(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File not found.")
        return

    plt.figure(figsize=(8, 6))

    # 1. Combined Model (Dynamic)
    plt.plot(df['epoch'], df['pooled_auc_combined'],
             label='Combined (Proposed)',
             color=STYLE_COLORS["Combined"], linewidth=3)

    # 2. Baseline (Static/Max)
    # נניח שהבייסליין הוא המקסימום שהושג (או הממוצע, תלוי מה הגיוני בנתונים שלך)
    baseline_val = df['pooled_auc_baseline'].max()
    plt.axhline(y=baseline_val, color=STYLE_COLORS["Baseline"], linestyle='--', linewidth=2,
                label=f'Baseline (Static) = {baseline_val:.3f}')

    # 3. Selected Epoch Marker
    if selected_epoch in df['epoch'].values:
        sel_auc = df.loc[df['epoch'] == selected_epoch, 'pooled_auc_combined'].values[0]

        plt.axvline(x=selected_epoch, color='gray', linestyle=':', alpha=0.6)
        plt.scatter(selected_epoch, sel_auc, color=STYLE_COLORS["Highlight"], s=100, zorder=10,
                    edgecolors='white', linewidth=1.5)

        plt.text(selected_epoch, sel_auc + 0.005, f" Epoch {selected_epoch}",
                 color=STYLE_COLORS["Highlight"], fontweight='bold', ha='left', va='bottom')

    plt.xlabel('Training Epochs')
    plt.ylabel('Validation AUC (Pooled)')
    plt.title('Training Dynamics vs. Baseline', fontweight='bold')

    # גבולות Y דינמיים אבל הגיוניים
    y_min = min(df['pooled_auc_combined'].min(), baseline_val) - 0.02
    y_max = max(df['pooled_auc_combined'].max(), baseline_val) + 0.02
    plt.ylim(y_min, y_max)
    plt.xlim(left=0)

    plt.legend(loc='lower right', frameon=False)
    sns.despine()
    plt.tight_layout()
    plt.savefig(csv_dir/'learning_curve_with_baseline.png', dpi=300, bbox_inches='tight')
    print("Saved: learning_curve_with_baseline.png")
    plt.close()


# ==========================================
#   Function 3: Synergy Gap
# ==========================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==========================================
#   GLOBAL STYLE SETTINGS
# ==========================================
sns.set_style("ticks", {"axes.grid": True, "grid.linestyle": ":", "grid.alpha": 0.5})
sns.set_context("paper", font_scale=1.5)  # הגדלתי קצת את הפונט לקריאות

STYLE_COLORS = {
    "Baseline": "#34495e",  # Dark Grey/Blue
    "Combined": "#e41a1c",  # Red
    "Embedding-only": "#4daf4a",  # Green
    "Highlight": "#ff7f00"  # Orange
}


def plot_synergy_gap_polished(csv_dir, csv_path, selected_epoch=76):
    """
    Dual-axis plot: Absolute AUC (Left) vs Delta AUC (Right).
    Legend outside. Text label moved to the LEFT to avoid axis overlap.
    """
    print(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("Error: File not found.")
        return

    # נתונים
    combined_series = df['pooled_auc_combined']
    emb_series = df['pooled_auc_embedding_only']
    baseline_series = df['pooled_auc_baseline']

    # חישוב רפרנס לבייסליין
    baseline_ref = baseline_series.mean()

    # יצירת הפיגר
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # ==========================
    # ציר שמאל: ערכים מוחלטים
    # ==========================

    # 1. קו הבייסליין
    l1, = ax1.plot(df['epoch'], baseline_series, color=STYLE_COLORS["Baseline"],
             linestyle='-', linewidth=1.5, label='Baseline (Reference)')

    # 2. Combined (Gain)
    l2, = ax1.plot(df['epoch'], combined_series, color=STYLE_COLORS["Combined"],
             linewidth=2.5, label='Combined (Model)')

    # 3. Embedding (Gap)
    l3, = ax1.plot(df['epoch'], emb_series, color=STYLE_COLORS["Embedding-only"],
             linewidth=2.5, linestyle='--', label='Embedding Only')

    # מילוי שטחים
    ax1.fill_between(df['epoch'], baseline_series, combined_series,
                     where=(combined_series >= baseline_series),
                     color=STYLE_COLORS["Combined"], alpha=0.1)

    ax1.fill_between(df['epoch'], baseline_series, emb_series,
                     color=STYLE_COLORS["Embedding-only"], alpha=0.05)

    ax1.set_xlabel('Training Epochs', fontweight='bold')
    ax1.set_ylabel('Absolute Pooled AUC', color='#333333', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#333333')
    ax1.grid(True, linestyle=':', alpha=0.5)

    # ==========================
    # ציר ימין: דלתא (מסונכרן)
    # ==========================
    ax2 = ax1.twinx()

    # חישוב גבולות וסנכרון
    y_min_data = min(emb_series.min(), baseline_series.min())
    y_max_data = max(combined_series.max(), baseline_series.max())
    padding = (y_max_data - y_min_data) * 0.20

    ylim_bottom = y_min_data - padding
    ylim_top = y_max_data + padding

    ax1.set_ylim(ylim_bottom, ylim_top)
    ax2.set_ylim(ylim_bottom - baseline_ref, ylim_top - baseline_ref)

    ax2.set_ylabel('Δ AUC (Relative to Baseline)', color='#555555', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#555555')

    ax2.axhline(0, color='gray', linewidth=0.8, alpha=0.4, zorder=0)

    # ==========================
    # סימונים (Annotations) - התיקון כאן
    # ==========================
    if selected_epoch in df['epoch'].values:
        val_comb = combined_series[df['epoch'] == selected_epoch].values[0]
        val_emb = emb_series[df['epoch'] == selected_epoch].values[0]

        # קו אנכי עדין
        ax1.axvline(x=selected_epoch, color='gray', linestyle=':', alpha=0.5)

        # נקודות
        ax1.scatter(selected_epoch, val_comb, color=STYLE_COLORS["Combined"], s=90, zorder=10, edgecolors='white', linewidth=1.5)
        ax1.scatter(selected_epoch, val_emb, color=STYLE_COLORS["Embedding-only"], s=90, zorder=10, edgecolors='white', linewidth=1.5)

        # חץ Synergy
        ax1.annotate('',
                     xy=(selected_epoch, val_comb),
                     xytext=(selected_epoch, val_emb),
                     arrowprops=dict(arrowstyle='<|-|>', color='#444444', lw=1.5, mutation_scale=15))

        # --- התיקון: הטקסט זז שמאלה ---
        mid_y = (val_comb + val_emb) / 2

        # selected_epoch - 2: מזיז את הטקסט 2 אפוקים שמאלה
        # ha='right': מיישר את הטקסט לימין, כך שהוא "גדל" לכיוון פנים הגרף ולא החוצה
        ax1.text(selected_epoch - 2, mid_y, 'Synergy\nGap',
                 horizontalalignment='right',  # יישור לימין (חשוב!)
                 verticalalignment='center',
                 color='#444444', fontweight='bold', fontsize=11,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8)) # רקע לבן אטום יותר

    # ==========================
    # Legend
    # ==========================
    legend_elements = [l1, l2, l3]
    ax1.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 1.02),
               ncol=3, frameon=False, fontsize=11)

    sns.despine(ax=ax1, top=True, right=False)
    sns.despine(ax=ax2, top=True, left=True, right=False)

    plt.title('Synergy Analysis: Gain over Baseline', fontweight='bold', y=1.12)
    plt.tight_layout()

    out_file = 'synergy_gap_polished.png'
    plt.savefig(csv_dir /out_file, dpi=300, bbox_inches='tight')
    print(f"Saved clean plot: {out_file}")
    plt.show()
    plt.close()


# ==========================================
#   MAIN EXECUTION
# ==========================================

csv_dir = Path('/mnt/new_groups/vaksler_group/Efrat/Results/experiment_runs_featuers_new_try/run_epochs_80.0_lr_0.006_din_128.0_dout_64.0_bs_64.0')
plot_learning_dynamics(csv_dir,csv_dir / "three_models_epoch_auc_table.csv", selected_epoch=76)
plot_synergy_gap_polished(csv_dir,csv_dir /"three_models_epoch_auc_table.csv", selected_epoch=76)
plot_delta_auc_distribution(csv_dir, csv_dir / "delong_results.csv")