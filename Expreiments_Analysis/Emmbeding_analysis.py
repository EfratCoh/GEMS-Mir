import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
from pathlib import Path

# def visualize_embeddings(csv_path, title="Embedding Visualization"):
#     # 1. טעינת הנתונים
#     df = pd.read_csv(csv_path)
#
#     # 2. הפרדת ה-ID מהוקטורים
#     # נניח שהעמודה הראשונה היא ID_interaction והשאר הן ה-dimensions
#     ids = df['ID_interaction']
#     embeddings = df.drop(columns=['ID_interaction']).values
#
#     # 3. יצירת לייבלים לפי השם (לפי הלוגיקה של הקוד הקודם שלך)
#     # חיובי: darnell_human, שלילי: NPS (Negative)
#     labels = ids.apply(lambda x: 'Negative' if 'NPS' in x else 'Positive')
#
#     print(f"Total samples: {len(df)} | Positive: {sum(labels == 'Positive')} | Negative: {sum(labels == 'Negative')}")
#
#     # 4. הרצת t-SNE (הורדת ממד ל-2)
#     # perplexity משפיע על מבנה הצבירים, בדרך כלל בין 5 ל-50
#     tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
#     embeddings_2d = tsne.fit_transform(embeddings)
#
#     # 5. יצירת ה-DataFrame לציור
#     viz_df = pd.DataFrame({
#         'x': embeddings_2d[:, 0],
#         'y': embeddings_2d[:, 1],
#         'Label': labels
#     })
#
#     # 6. ציור הגרף
#     plt.figure(figsize=(10, 7))
#     sns.scatterplot(
#         data=viz_df,
#         x='x', y='y',
#         hue='Label',
#         palette={'Positive': 'blue', 'Negative': 'red'},
#         alpha=0.6,
#         s=50
#     )
#
#     plt.title(title)
#     plt.xlabel('t-SNE dimension 1')
#     plt.ylabel('t-SNE dimension 2')
#     plt.grid(True, alpha=0.3)
#     plt.legend(title='Interaction Type')
#     plt.show()
#


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def visualize_embeddings(csv_path, title="Improved Embedding Visualization"):
    # 1. טעינת הנתונים
    df = pd.read_csv(csv_path)

    # 2. הפרדת ה-ID והנתונים
    # נניח שהעמודה הראשונה היא ה-ID והשאר הן ה-Embeddings
    # אם יש עמודות נוספות שאינן מספרים (כמו שם הקובץ וכו'), צריך להסיר גם אותן
    ids = df['ID_interaction']

    # מוודאים שלוקחים רק את העמודות המספריות של הוקטור
    # כאן אני מניח שכל מה שאינו ID הוא דאטה. אם יש עמודות טקסט אחרות, צריך להסיר אותן עם drop
    embeddings = df.drop(columns=['ID_interaction'])

    # 3. יצירת לייבלים
    labels = ids.apply(lambda x: 'Negative' if 'NPS' in str(x) else 'Positive')

    print(f"Total samples: {len(df)} | Positive: {sum(labels == 'Positive')} | Negative: {sum(labels == 'Negative')}")

    # --- שלבי השיפור ---

    # 4. נרמול (Scaling) - קריטי להפרדה טובה!
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    # 5. PCA מקדים (Pre-processing)
    # t-SNE עובד טוב יותר כשיש פחות רעש. נוריד ל-50 מימדים ראשוניים שמסבירים את רוב השונות
    # אם יש לך פחות מ-50 מימדים במקור, אפשר לדלג על השלב הזה
    if embeddings_scaled.shape[1] > 50:
        pca = PCA(n_components=50)
        embeddings_pca = pca.fit_transform(embeddings_scaled)
    else:
        embeddings_pca = embeddings_scaled

    # 6. הרצת t-SNE עם פרמטרים משופרים
    # metric='cosine': מתאים יותר לוקטורים של שפה/ביולוגיה מאשר euclidean
    # perplexity: משחק עם הערך הזה (בין 30 ל-50) יכול לשנות את צורת הצבירים
    tsne = TSNE(
        n_components=2,
        perplexity=40,  # העליתי קצת, לפעמים נותן מבנה גלובלי טוב יותר
        early_exaggeration=12,  # ערך ברירת מחדל, אבל אפשר להגדיל כדי להרחיק צבירים (למשל 20)
        metric='cosine',  # שינוי קריטי! מרחק קוסינוס עדיף לרוב ל-Embeddings
        init='pca',
        learning_rate='auto',
        n_iter=1000,  # לוודא שיש מספיק איטרציות להתכנסות
        random_state=42,
        n_jobs=-1  # שימוש בכל הליבות להאצה
    )

    embeddings_2d = tsne.fit_transform(embeddings_pca)

    # 7. יצירת ה-DataFrame לציור
    viz_df = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'Label': labels
    })

    # 8. ציור הגרף
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=viz_df,
        x='x', y='y',
        hue='Label',
        style='Label',  # הוספתי צורות שונות ליתר בהירות
        palette={'Positive': '#2ecc71', 'Negative': '#e74c3c'},  # צבעים קונטרסטיים יותר (ירוק/אדום)
        alpha=0.7,  # שקיפות
        s=60,  # גודל נקודה
        edgecolor='w',  # מסגרת לבנה לנקודות להפרדה ויזואלית
        linewidth=0.5
    )

    plt.title(f"{title}\n(StandardScaler + PCA + Cosine t-SNE)", fontsize=14)
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.legend(title='Interaction Type', loc='best')
    plt.grid(True, alpha=0.2, linestyle='--')
    plt.tight_layout()
    plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import StandardScaler


def visualize_embeddings_umap(csv_path, title="UMAP Projection of Embeddings"):
    """
    Reads embedding data from CSV, applies UMAP with Cosine metric,
    and plots the result with clear separation between Positive and Negative samples.
    """

    # 1. טעינת הנתונים
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # 2. חילוץ ה-ID והנתונים
    # מניחים שהעמודה הראשונה היא ה-ID. אם השם שונה, שניה את 'ID_interaction'
    if 'ID_interaction' in df.columns:
        ids = df['ID_interaction']
        # הסרת עמודת ה-ID כדי להשאיר רק את הוקטורים המספריים
        embeddings = df.drop(columns=['ID_interaction'])
    else:
        # גיבוי: אם שם העמודה לא ידוע, לוקחים את העמודה הראשונה כ-ID
        ids = df.iloc[:, 0]
        embeddings = df.iloc[:, 1:]

    # 3. יצירת לייבלים (Ground Truth)
    # לוגיקה: אם המחרוזת 'NPS' מופיעה ב-ID זה שלילי, אחרת חיובי
    labels = ids.apply(lambda x: 'Negative' if 'NPS' in str(x) else 'Positive')

    # הדפסת סטטיסטיקה קצרה
    pos_count = sum(labels == 'Positive')
    neg_count = sum(labels == 'Negative')
    print(f"Data loaded. Positive: {pos_count} | Negative: {neg_count}")

    # 4. נרמול (Standard Scaling) - שלב קריטי!
    # UMAP עובד הרבה יותר טוב כשהנתונים מנורמלים (ממוצע 0, סטיית תקן 1)
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    # 5. הרצת UMAP
    print("Running UMAP...")
    reducer = umap.UMAP(
        n_neighbors=50,  # ברירת מחדל 15. ערך גבוה יותר (30-50) שומר יותר על המבנה הגלובלי ומפריד צבירים
        min_dist=0.1,  # כמה צפופות הנקודות בתוך הצביר (0.1 זה סטנדרטי)
        n_components=2,  # ממד היעד
        metric='cosine',  # קריטי: לאמבדינג עדיף מרחק קוסינוס ולא אוקלידי
        random_state=42  # לשחזור תוצאות
    )

    embedding_2d = reducer.fit_transform(embeddings_scaled)

    # 6. הכנת הדאטה לציור
    viz_df = pd.DataFrame({
        'UMAP 1': embedding_2d[:, 0],
        'UMAP 2': embedding_2d[:, 1],
        'Label': labels
    })

    # 7. ציור הגרף
    plt.figure(figsize=(12, 8))

    scatter = sns.scatterplot(
        data=viz_df,
        x='UMAP 1',
        y='UMAP 2',
        hue='Label',
        style='Label',  # מוסיף גם צורה שונה לכל קלאס (עיגול מול איקס)
        palette={'Positive': '#2ecc71', 'Negative': '#e74c3c'},  # ירוק ואדום בוהקים
        s=40,  # גודל הנקודות
        alpha=0.5,  # שקיפות קלה כדי לראות צפיפות
        edgecolor='w',  # מסגרת לבנה סביב הנקודות
        linewidth=0.5
    )

    plt.title(f"{title}\n(UMAP: metric=cosine, n_neighbors=30)", fontsize=15)
    plt.legend(title='Interaction Type', frameon=True, loc='best')
    plt.grid(True, alpha=0.15, linestyle='--')
    plt.tight_layout()

    print("Plot generated successfully.")
    plt.show()
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    # בדיקה: האם הוקטורים האלו ניתנים להפרדה לינארית?
    clf = LogisticRegression(max_iter=1000)
    scores = cross_val_score(clf, embeddings_scaled, labels, cv=5, scoring='accuracy')

    print(f"Linear Separation Accuracy: {scores.mean():.3f}")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score

    # 1. לינארי (כבר עשית)
    # log_reg = LogisticRegression(max_iter=1000)
    # print(f"Linear: {cross_val_score(log_reg, embeddings_scaled, labels, cv=5).mean():.3f}")

    # 2. Random Forest (לא לינארי - עץ החלטה)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_acc = cross_val_score(rf, embeddings_scaled, labels, cv=5).mean()
    print(f"Random Forest Accuracy: {rf_acc:.3f}")

    # 3. SVM (עם גרעין RBF - יודע לעקם את המרחב)
    svm = SVC(kernel='rbf')
    svm_acc = cross_val_score(svm, embeddings_scaled, labels, cv=5).mean()
    print(f"SVM (RBF) Accuracy: {svm_acc:.3f}")


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import umap
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def draw_confidence_ellipse(x, y, ax, n_std=2.0, **kwargs):
    cov = np.cov(x, y)

    # פירוק עצמי (alignment נכון)
    vals, vecs = np.linalg.eigh(cov)

    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    width, height = 2 * n_std * np.sqrt(vals)

    ellipse = Ellipse(
        (np.mean(x), np.mean(y)),
        width,
        height,
        angle=theta,
        fill=False,
        **kwargs
    )

    return ax.add_patch(ellipse)


def visualize_with_structure(csv_path):
    # --- 1. טעינה ועיבוד (כמו קודם) ---
    df = pd.read_csv(csv_path)
    if 'ID_interaction' in df.columns:
        ids = df['ID_interaction']
        embeddings = df.drop(columns=['ID_interaction'])
    else:
        ids = df.iloc[:, 0]
        embeddings = df.iloc[:, 1:]

    labels = ids.apply(lambda x: 'Negative' if 'NPS' in str(x) else 'Positive')

    # נרמול
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    # UMAP
    reducer = umap.UMAP(n_neighbors=100, min_dist=0.1, n_components=2, metric='cosine', random_state=42)
    embedding_2d = reducer.fit_transform(embeddings_scaled)

    # יצירת DataFrame לציור
    viz_df = pd.DataFrame({
        'x': embedding_2d[:, 0],
        'y': embedding_2d[:, 1],
        'Label': labels
    })

    # --- 2. ציור הגרף המשודרג ---
    fig, ax = plt.subplots(figsize=(10, 8))

    # א. ציור הנקודות (קצת שקופות כדי לא להסתיר את האליפסות)
    sns.scatterplot(data=viz_df, x='x', y='y', hue='Label', style='Label',
                    palette={'Positive': '#2ecc71', 'Negative': '#e74c3c'},
                    s=30, alpha=0.4, ax=ax, legend=False)

    # ב. לולאה לציור המרכז והאליפסה לכל קבוצה
    colors = {'Positive': '#2ecc71', 'Negative': '#e74c3c'}

    for label, color in colors.items():
        subset = viz_df[viz_df['Label'] == label]

        # 1. ציור אליפסה (Confidence Ellipse)
        # edgecolor=color -> צבע המסגרת לפי הקבוצה
        # linestyle='--' -> קו מקווקו
        draw_confidence_ellipse(subset['x'], subset['y'], ax, n_std=2.0,
                                edgecolor=color, linestyle='--', linewidth=2, label=f'{label} (95% CI)')

        # 2. ציור מרכז המסה (Centroid) - ה-X הגדול
        center_x = subset['x'].mean()
        center_y = subset['y'].mean()
        ax.scatter(center_x, center_y, color='black', marker='X', s=200, zorder=10,
                   edgecolor='white', linewidth=1.5, label=f'{label} Centroid')
        ax.text(center_x, center_y, f'  {label}\n  Center', fontsize=12, fontweight='bold', color='black')

    # ג. אופציה נוספת: KDE (קווי גובה) - אם רוצים מראה "טופוגרפי" במקום אליפסה
    # sns.kdeplot(data=viz_df, x='x', y='y', hue='Label', palette=colors, levels=5, thresh=0.2, ax=ax, linewidths=1.5)

    plt.title("UMAP Projection with Class Centroids & Confidence Ellipses", fontsize=15)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")

    # יצירת מקרא מותאם אישית (כדי לא לכפול את הנקודות והאליפסות)
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='#2ecc71', lw=4),
                    Line2D([0], [0], color='#e74c3c', lw=4),
                    Line2D([0], [0], marker='X', color='w', markerfacecolor='k', markersize=10)]
    ax.legend(custom_lines, ['Positive', 'Negative', 'Centroid (Center of Mass)'], loc='best')

    plt.grid(True, alpha=0.2, linestyle='--')
    plt.show()


# visualize_with_structure('your_file.csv')
base_dir =Path("/mnt/new_groups/vaksler_group/Efrat/Results/experiment_runs_featuers_new_try/run_epochs_120.0_lr_0.001_din_64.0_dout_256.0_bs_32.0/Data_embedding")
# visualize_embeddings(base_dir / "fold1/epoch_48/test_embeddings.csv", title="Embeddings Space - Epoch 48")
# visualize_embeddings_umap(base_dir / "fold1/epoch_80/test_embeddings.csv")
# visualize_with_structure(base_dir / "fold1/epoch_80/test_embeddings.csv")
#


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def prepare_data(df):
    """
    פונקציית עזר לסידור הדאטה:
    מחזירה את X (הפיצ'רים) ואת y (הלייבלים)
    """
    # 1. זיהוי עמודת ה-ID
    if 'ID_interaction' in df.columns:
        ids = df['ID_interaction']
        X = df.drop(columns=['ID_interaction'])
    else:
        # ברירת מחדל: העמודה הראשונה היא ID
        ids = df.iloc[:, 0]
        X = df.iloc[:, 1:]

    # 2. יצירת לייבלים (1=Positive, 0=Negative)
    # לוגיקה: אם כתוב NPS זה שלילי, אחרת חיובי
    y = ids.apply(lambda x: 0 if 'NPS' in str(x) else 1)

    return X, y


def train_and_evaluate(train_csv_path, test_csv_path):
    print(f"--- Loading Data ---")
    df_train = pd.read_csv(train_csv_path)
    df_test = pd.read_csv(test_csv_path)

    # הכנת הנתונים
    X_train, y_train = prepare_data(df_train)
    X_test, y_test = prepare_data(df_test)

    print(f"Train Shape: {X_train.shape} | Positive: {sum(y_train == 1)}, Negative: {sum(y_train == 0)}")
    print(f"Test Shape:  {X_test.shape} | Positive: {sum(y_test == 1)}, Negative: {sum(y_test == 0)}")
    print("-" * 50)

    # # 3. נרמול (Scaling) - קריטי!
    # # לומדים את הפרמטרים (ממוצע וסטיית תקן) רק מהאימון!
    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    #
    # # את הטסט רק מנרמלים לפי מה שלמדנו מהאימון (אסור לעשות fit על הטסט!)
    # X_test_scaled = scaler.transform(X_test)

    # ==========================================
    # מודל 1: Random Forest
    # ==========================================
    X_train_scaled = X_train
    X_test_scaled = X_test
    print("\n>>> 1. Random Forest Classifier")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)

    # חיזוי על סט המבחן
    y_pred_rf = rf.predict(X_test_scaled)

    print(f"Test Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_rf))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_rf, target_names=['Negative', 'Positive']))

    # ==========================================
    # מודל 2: SVM (RBF Kernel)
    # ==========================================
    print("\n>>> 2. SVM (Support Vector Machine)")
    svm = SVC(kernel='rbf', probability=True, random_state=42)
    svm.fit(X_train_scaled, y_train)

    # חיזוי על סט המבחן
    y_pred_svm = svm.predict(X_test_scaled)

    print(f"Test Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_svm))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_svm, target_names=['Negative', 'Positive']))

# --- דוגמה להפעלה ---
train_and_evaluate(base_dir / "fold1/epoch_119/train_embeddings.csv", base_dir / "fold1/epoch_119/test_embeddings.csv")