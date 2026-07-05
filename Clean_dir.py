# import os
# from pathlib import Path
# import os
# import shutil
# from pathlib import Path
#
# def clean_embedding_dir(base_dir):
#     base_path = Path(base_dir)
#
#     if not base_path.exists():
#         print(f"The directory {base_dir} does not exist.")
#         return
#
#     # Iterate over all subdirectories
#     for sub_dir in base_path.glob('**/'):  # '**/' matches all subdirectories recursively
#         # Iterate over all files in the current subdirectory
#         for file_path in sub_dir.iterdir():
#             if file_path.is_file():
#                 try:
#                     file_path.unlink()  # Remove the file
#                     print(f"Deleted file: {file_path}")
#                 except Exception as e:
#                     print(f"Failed to delete {file_path}: {e}")
#
#     print(f"Cleaned all files in subdirectories under {base_dir}.")
#
# # Specify the directory
# # a_dir = "/home/efrco/PHD/Goal_One/Data_Graph_Matrix/NPS_darnell_human_dataset"
# # a_dir = "/home/efrco/PHD/Goal_One/Data_Folds/"
# # a_dir = "/groups/vaksler_group/Efrat/Results/experiment_runs_16_featuers/run_epochs_40.0_lr_0.006_din_128.0_dout_64.0_bs_32.0/"
# # a_dir = "/groups/vaksler_group/Efrat/Results/experiment_runs_featuers/"
#
# a_dir = Path("/mnt/new_groups/vaksler_group/Efrat/Results/experiment_runs_featuers_new_try/run_epochs_80.0_lr_0.006_din_128.0_dout_64.0_bs_64.0/fold5/")
# shutil.rmtree(Path(a_dir))
#
# # # Run the cleaning function
# # clean_embedding_dir(a_dir)
# import os
# import shutil
# from pathlib import Path
#
# # base_dir = Path("/home/efrco/PHD/Goal_One/Data_Graph_Matrix/NPS_darnell_human_dataset/")
# # base_dir = Path("/home/efrco/PHD/Goal_One/Results/experiment_runs/")
# # base_dir = Path("/groups/vaksler_group/Efrat/")
# # clean_embedding_dir(base_dir)
# # לולאה על כל תתי התיקיות
# #fdkd
# print("ddd")
# for exp_dir in base_dir.glob("*"):  # ניסויים שונים
#     if not exp_dir.is_dir():
#         continue
#
#     result_dir = exp_dir / "Results"
#     if result_dir.exists() and result_dir.is_dir():
#         if not any(result_dir.iterdir()):  # בודק אם ריקה
#             print(f"Deleting experiment folder: {exp_dir}")
#             shutil.rmtree(exp_dir)  # מוחק את כל תיקיית הניסוי
#
#
# from pathlib import Path
# for p in base_dir.glob("*.csv"):
#     p.unlink()
#

from consts.global_consts import NEGATIVE_DATA_PATH, POSITIVE_PATH_FEATUERS
from sklearn.model_selection import StratifiedKFold
from pathlib import Path


pos_dataset_darnell = POSITIVE_PATH_FEATUERS/ "darnell_human_ViennaDuplex_75nt_fragment_clean_features.csv"
neg_dataset_NPS_CLASH = NEGATIVE_DATA_PATH / "non_overlapping_sites/non_overlapping_sites_darnell_human_ViennaDuplex_75nt_fragment_clean_negative_features.csv"

import pandas as pd
from pathlib import Path

# === עדכני את הנתיבים לכאלו שאצלך במחשב ===
FOLDS_BASE_PATH = Path("/home/efrco/PHD/Goal_One/Data_Folds")  # נתיב תיקיית הפולדים

POS_FILE = POSITIVE_PATH_FEATUERS/ "darnell_human_ViennaDuplex_75nt_fragment_clean_features.csv"
NEG_FILE = NEGATIVE_DATA_PATH / "non_overlapping_sites/non_overlapping_sites_darnell_human_ViennaDuplex_75nt_fragment_clean_negative_features.csv"


# def main():
#     print("🔍 בודק Data Leakage ברמת הזוג הביולוגי האמיתי (Gene_ID + miRNA ID)...")
#
#     # 1. קריאת הקבצים המקוריים ליצירת מילון תרגום
#     id_to_bio_pair = {}
#
#     for filepath in [POS_FILE, NEG_FILE]:
#         if not filepath.exists():
#             print(f"⚠️ הקובץ לא נמצא, מדלג עליו (תבדקי את הנתיב): {filepath}")
#             continue
#
#         df_orig = pd.read_csv(filepath)
#
#         # נוודא שהעמודות קיימות בקובץ
#         if 'ID_interaction' in df_orig.columns and 'miRNA ID' in df_orig.columns and 'Gene_ID' in df_orig.columns:
#             for _, row in df_orig.iterrows():
#                 inter_id = str(row['ID_interaction'])
#                 mirna = str(row['miRNA sequence'])
#                 gene = str(row['fragment'])
#
#                 # חיבור ה-miRNA והגן לזוג אחד ייחודי
#                 id_to_bio_pair[inter_id] = f"{mirna}_{gene}"
#         else:
#             print(f"⚠️ חסרות עמודות קריטיות בקובץ {filepath.name}")
#
#     if not id_to_bio_pair:
#         print("❌ לא הצלחתי לבנות מילון. אנא ודאי שהעמודות כתובות במדויק.")
#         return
#
#     print(f"✅ מילון תרגום מוכן. נטענו {len(id_to_bio_pair)} אינטראקציות מהקבצים המקוריים.")
#     print("-" * 50)
#
#     # 2. בדיקת הפולדים
#     total_leakages = []
#
#     for f in range(1, 11):
#         fold_name = f"fold{f}"
#         train_path = FOLDS_BASE_PATH / fold_name / f"{fold_name}_train.csv"
#         test_path = FOLDS_BASE_PATH / fold_name / f"{fold_name}_test.csv"
#
#         if not train_path.exists() or not test_path.exists():
#             continue
#
#         df_train = pd.read_csv(train_path)
#         df_test = pd.read_csv(test_path)
#
#         train_pairs = set()
#         test_pairs = set()
#
#         # המרה של המזהים ב-Train לשמות ביולוגיים
#         for val in df_train['ID_interaction']:
#             val_str = str(val)
#             if val_str in id_to_bio_pair:
#                 train_pairs.add(id_to_bio_pair[val_str])
#
#         # המרה של המזהים ב-Test לשמות ביולוגיים
#         for val in df_test['ID_interaction']:
#             val_str = str(val)
#             if val_str in id_to_bio_pair:
#                 test_pairs.add(id_to_bio_pair[val_str])
#
#         if len(test_pairs) == 0:
#             print(f"Fold {f}: לא נמצאו התאמות במילון עבור קבוצת הטסט (בדקי את תאימות השמות של ID_interaction).")
#             continue
#
#         leaked = train_pairs.intersection(test_pairs)
#         leakage_pct = (len(leaked) / len(test_pairs)) * 100
#         total_leakages.append(leakage_pct)
#
#         print(f"Fold {f}: {len(leaked)} leaked pairs out of {len(test_pairs)} unique test pairs ({leakage_pct:.2f}%)")
#
#     if total_leakages:
#         avg = sum(total_leakages) / len(total_leakages)
#         print("-" * 50)
#         print(f"🏁 בדיקה הושלמה! אחוז הזליגה האמיתי בממוצע: {avg:.2f}%")
#
#
# if __name__ == "__main__":
#     main()

import pandas as pd
from pathlib import Path

# === הנתיבים ===
FOLDS_BASE_PATH = Path("/home/efrco/PHD/Goal_One/Data_Folds")
POS_FILE = POSITIVE_PATH_FEATUERS/ "darnell_human_ViennaDuplex_75nt_fragment_clean_features.csv"


def main():
    print("🔍 בודק Data Leakage עבור הסט החיובי בלבד (Positive Data)...")

    # 1. קריאת הקובץ החיובי ליצירת מילון תרגום
    id_to_bio_pair = {}

    if not POS_FILE.exists():
        print(f"⚠️ הקובץ לא נמצא (תבדקי את הנתיב): {POS_FILE}")
        return

    # טעינה של הקובץ החיובי בלבד
    df_pos = pd.read_csv(POS_FILE, low_memory=False)

    if 'ID_interaction' in df_pos.columns and 'miRNA ID' in df_pos.columns and 'Gene_ID' in df_pos.columns:
        for _, row in df_pos.iterrows():
            inter_id = str(row['ID_interaction'])
            mirna = str(row["miRNA ID"])
            gene = str(row['Gene_ID'])
            id_to_bio_pair[inter_id] = f"{mirna}_{gene}"
    else:
        print("⚠️ חסרות עמודות קריטיות בקובץ החיובי.")
        return

    print(f"✅ נטענו {len(id_to_bio_pair)} אינטראקציות חיוביות למילון התרגום.")
    print("-" * 50)

    # 2. בדיקת הפולדים (הסקריפט יסנן אוטומטית החוצה את השליליים כי הם לא במילון)
    total_leakages = []

    for f in range(1, 11):
        fold_name = f"fold{f}"
        train_path = FOLDS_BASE_PATH / fold_name / f"{fold_name}_train.csv"
        test_path = FOLDS_BASE_PATH / fold_name / f"{fold_name}_test.csv"

        if not train_path.exists() or not test_path.exists():
            continue

        df_train = pd.read_csv(train_path, low_memory=False)
        df_test = pd.read_csv(test_path, low_memory=False)

        train_pairs = set()
        test_pairs = set()

        # המרה של המזהים ב-Train (רק החיוביים ייכנסו לקבוצה)
        for val in df_train['ID_interaction']:
            val_str = str(val)
            if val_str in id_to_bio_pair:
                train_pairs.add(id_to_bio_pair[val_str])

        # המרה של המזהים ב-Test (רק החיוביים ייכנסו לקבוצה)
        for val in df_test['ID_interaction']:
            val_str = str(val)
            if val_str in id_to_bio_pair:
                test_pairs.add(id_to_bio_pair[val_str])

        if len(test_pairs) == 0:
            print(f"Fold {f}: לא נמצאו אינטראקציות חיוביות בטסט.")
            continue

        leaked = train_pairs.intersection(test_pairs)
        leakage_pct = (len(leaked) / len(test_pairs)) * 100
        total_leakages.append(leakage_pct)

        print(
            f"Fold {f}: {len(leaked)} leaked POSITIVE pairs out of {len(test_pairs)} unique POSITIVE test pairs ({leakage_pct:.2f}%)")

    if total_leakages:
        avg = sum(total_leakages) / len(total_leakages)
        print("-" * 50)
        print(f"🏁 בדיקה הושלמה! אחוז הזליגה בסט החיובי בלבד: {avg:.2f}%")


# if __name__ == "__main__":
#     main()

import pandas as pd
from pathlib import Path

# === הגדרת נתיבים ===
FOLDS_BASE_PATH = Path("/home/efrco/PHD/Goal_One/Data_Folds")
POS_FILE = POSITIVE_PATH_FEATUERS/ "darnell_human_ViennaDuplex_75nt_fragment_clean_features.csv"

# סף להגדרת חפיפה מרחבית (מספר נוקלאוטידים רציפים זהים)
OVERLAP_THRESHOLD = 75


def find_lcs_length(s1, s2):
    """מוצא את אורך תת-המחרוזת הרציפה הארוכה ביותר בין שני רצפים"""
    if not isinstance(s1, str) or not isinstance(s2, str):
        return 0
    m = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]
    longest = 0
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            if s1[i - 1] == s2[j - 1]:
                m[i][j] = m[i - 1][j - 1] + 1
                if m[i][j] > longest:
                    longest = m[i][j]
            else:
                m[i][j] = 0
    return longest


def main():
    print(f"🔍 מתחיל בדיקת חפיפה מרחבית (Spatial Overlap) עבור הסט החיובי...")
    print(f"📏 סף חפיפה מוגדר: {OVERLAP_THRESHOLD} נוקלאוטידים רציפים זהים.\n")

    if not POS_FILE.exists():
        print(f"⚠️ קובץ המקור לא נמצא: {POS_FILE}")
        return

    # 1. טעינת הדאטה החיובי ובניית מילון מיפוי
    df_pos = pd.read_csv(POS_FILE, low_memory=False)

    # מיפוי מזהה אינטראקציה לפרטים הביולוגיים שלה
    interaction_details = {}
    for _, row in df_pos.iterrows():
        inter_id = str(row['ID_interaction'])
        interaction_details[inter_id] = {
            'bio_pair': f"{row['miRNA ID']}_{row['Gene_ID']}",
            'fragment': str(row['fragment'])
        }

    print(f"✅ נטענו {len(interaction_details)} אינטראקציות חיוביות ממילון המקור.")
    print("-" * 60)

    # 2. מעבר על הפולדים ובדיקת חפיפות בתוך הזוגות המודלפים ברמת ה-ID
    for f in range(1, 11):
        fold_name = f"fold{f}"
        train_path = FOLDS_BASE_PATH / fold_name / f"{fold_name}_train.csv"
        test_path = FOLDS_BASE_PATH / fold_name / f"{fold_name}_test.csv"

        if not train_path.exists() or not test_path.exists():
            continue

        df_train = pd.read_csv(train_path, low_memory=False)
        df_test = pd.read_csv(test_path, low_memory=False)

        # קיבוץ רצפי פרגמנטים לפי הזוג הביולוגי (miRNA_Gene) בטריין
        train_bio_groups = {}
        for val in df_train['ID_interaction']:
            val_str = str(val)
            if val_str in interaction_details:
                details = interaction_details[val_str]
                pair = details['bio_pair']
                frag = details['fragment']
                if pair not in train_bio_groups:
                    train_bio_groups[pair] = []
                train_bio_groups[pair].append(frag)

        # בדיקת הדגימות בטסט מול הטריין
        total_test_pos_instances = 0
        spatial_overlap_count = 0
        leaked_by_id_count = 0

        for val in df_test['ID_interaction']:
            val_str = str(val)
            if val_str in interaction_details:
                total_test_pos_instances += 1
                details = interaction_details[val_str]
                pair = details['bio_pair']
                test_frag = details['fragment']

                # אם הזוג הביולוגי הזה קיים גם בטריין (זוהה קודם כזליגה של כ-5%)
                if pair in train_bio_groups:
                    leaked_by_id_count += 1

                    # נבדוק האם הפרגמנט מהטסט חופף פיזית עם *אחד* מהפרגמנטים של אותו זוג בטריין
                    has_overlap = False
                    for train_frag in train_bio_groups[pair]:
                        lcs_len = find_lcs_length(test_frag, train_frag)
                        if lcs_len >= OVERLAP_THRESHOLD:
                            has_overlap = True
                            break  # מספיק חפיפה עם פרגמנט אחד בטריין

                    if has_overlap:
                        spatial_overlap_count += 1

        overlap_pct_from_leaked = (spatial_overlap_count / leaked_by_id_count * 100) if leaked_by_id_count > 0 else 0
        overlap_pct_from_total = (
                    spatial_overlap_count / total_test_pos_instances * 100) if total_test_pos_instances > 0 else 0

        print(f"==> Fold {f}:")
        print(f"    - סה\"כ דגימות חיוביות בטסט: {total_test_pos_instances}")
        print(f"    - זוגות המודלפים לפי ID (כמו שמצאת קודם): {leaked_by_id_count}")
        print(f"    - מתוכם, מספר הפרגמנטים עם חפיפה מרחבית אמיתית: {spatial_overlap_count}")
        print(f"    - אחוז חפיפה מתוך הזליגה הכללית: {overlap_pct_from_leaked:.2f}%")
        print(f"    - אחוז חפיפה מתוך סך הסט החיובי בטסט: {overlap_pct_from_total:.2f}%")
        print("-" * 60)

#
# if __name__ == "__main__":
#     main()

import pandas as pd
from pathlib import Path

# === הגדרת נתיבים ===
FOLDS_BASE_PATH = Path("/home/efrco/PHD/Goal_One/Data_Folds")
POS_FILE = POSITIVE_PATH_FEATUERS/ "darnell_human_ViennaDuplex_75nt_fragment_clean_features.csv"
import pandas as pd
from pathlib import Path
import pandas as pd
from pathlib import Path
from difflib import SequenceMatcher


FRAGMENT_OVERLAP_THRESHOLD = 25


def get_longest_common_substring_fast(s1, s2):
    """מוצא את תת-המחרוזת הרציפה הארוכה ביותר בצורה סופר-מהירה"""
    if not isinstance(s1, str) or not isinstance(s2, str):
        return ""
    match = SequenceMatcher(None, s1, s2).find_longest_match(0, len(s1), 0, len(s2))
    return s1[match.a: match.a + match.size]


def main():
    print(f"🔍 מתחיל בדיקה תלת-שלבית מחמירה:")
    print(f"1️⃣ סינון ראשוני: אותו רצף miRNA בדיוק בטריין ובטסט.")
    print(f"2️⃣ שלב א': חפיפת פרגמנטים מעל {FRAGMENT_OVERLAP_THRESHOLD} נוקלאוטידים.")
    print(f"3️⃣ שלב ב' (החידוד שלך): *גם* ה-Site של הטסט ו*גם* ה-Site של הטריין חייבים ליפול בתוך אזור החפיפה!\n")

    if not POS_FILE.exists():
        print(f"⚠️ קובץ המקור לא נמצא: {POS_FILE}")
        return

    df_pos = pd.read_csv(POS_FILE, low_memory=False)

    if 'site' not in df_pos.columns or 'miRNA sequence' not in df_pos.columns:
        print("⚠️ חסרות עמודות 'site' או 'miRNA sequence' בקובץ המקור!")
        return

    # מיפוי מזהה אינטראקציה לנתונים שלה
    interaction_details = {}
    for _, row in df_pos.iterrows():
        inter_id = str(row['ID_interaction'])
        interaction_details[inter_id] = {
            'mirna_seq': str(row['miRNA sequence']).upper().strip(),
            'fragment': str(row['fragment']).upper().strip(),
            'site': str(row['site']).upper().strip()
        }

    print(f"✅ נטענו {len(interaction_details)} אינטראקציות חיוביות.")
    print("-" * 60)

    for f in range(1, 11):
        fold_name = f"fold{f}"
        train_path = FOLDS_BASE_PATH / fold_name / f"{fold_name}_train.csv"
        test_path = FOLDS_BASE_PATH / fold_name / f"{fold_name}_test.csv"

        if not train_path.exists() or not test_path.exists():
            continue

        df_train = pd.read_csv(train_path, low_memory=False)
        df_test = pd.read_csv(test_path, low_memory=False)

        # בניית מילון טריין שבו המפתח הוא רצף ה-miRNA
        train_by_mirna = {}
        for val in df_train['ID_interaction']:
            val_str = str(val)
            if val_str in interaction_details:
                mirna = interaction_details[val_str]['mirna_seq']
                if mirna not in train_by_mirna:
                    train_by_mirna[mirna] = []
                train_by_mirna[mirna].append(interaction_details[val_str])

        total_test_pos_instances = 0
        total_fragment_overlaps = 0
        total_site_overlaps = 0

        for val in df_test['ID_interaction']:
            val_str = str(val)
            if val_str in interaction_details:
                total_test_pos_instances += 1
                details = interaction_details[val_str]
                test_mirna = details['mirna_seq']
                test_frag = details['fragment']
                test_site = details['site']

                passed_stage_1 = False
                passed_stage_2 = False

                if test_mirna in train_by_mirna:
                    for train_data in train_by_mirna[test_mirna]:
                        train_frag = train_data['fragment']
                        train_site = train_data['site']  # <--- הבאנו גם את ה-Site של הטריין

                        overlap_string = get_longest_common_substring_fast(train_frag, test_frag)

                        if len(overlap_string) >= FRAGMENT_OVERLAP_THRESHOLD:
                            passed_stage_1 = True

                            # התנאי המחמיר שלך: *גם* הטסט ו*גם* הטריין צריכים להיות בתוך החפיפה
                            if (len(test_site) > 5 and test_site in overlap_string) and \
                                    (len(train_site) > 5 and train_site in overlap_string and test_site == train_site):
                                passed_stage_2 = True
                                break

                if passed_stage_1:
                    total_fragment_overlaps += 1
                if passed_stage_2:
                    total_site_overlaps += 1

        pct_frag = (total_fragment_overlaps / total_test_pos_instances * 100) if total_test_pos_instances > 0 else 0
        pct_site = (total_site_overlaps / total_test_pos_instances * 100) if total_test_pos_instances > 0 else 0

        print(f"==> Fold {f}:")
        print(f"    - סה\"כ דגימות חיוביות בטסט: {total_test_pos_instances}")
        print(
            f"    - פרגמנטים עם חפיפה מול אותו miRNA (מעל {FRAGMENT_OVERLAP_THRESHOLD}nt): {total_fragment_overlaps} ({pct_frag:.2f}%)")
        print(
            f"    - 🎯 מתוכם, *שני האתרים* מוכלים במדויק בתוך החפיפה (זליגה פונקציונלית טהורה): {total_site_overlaps} ({pct_site:.2f}%)")
        print("-" * 60)


# if __name__ == "__main__":
#     main()

import pandas as pd
from pathlib import Path
from difflib import SequenceMatcher

# === הגדרת נתיבים ===
FRAGMENT_OVERLAP_THRESHOLD = 10
import pandas as pd
from pathlib import Path
from difflib import SequenceMatcher
from collections import defaultdict

# === הגדרת נתיבים ===
OUTPUT_GROUPS_FILE = Path("/home/efrco/PHD/Goal_One/POSITIVE_PATH_FEATUERS/interaction_groups.csv")
import pandas as pd
from pathlib import Path
from difflib import SequenceMatcher
from collections import defaultdict

# === הגדרת נתיבים ===
FRAGMENT_OVERLAP_THRESHOLD = 10


def get_longest_common_substring_fast(s1, s2):
    """מוצא את תת-המחרוזת הרציפה הארוכה ביותר"""
    if not isinstance(s1, str) or not isinstance(s2, str):
        return ""
    match = SequenceMatcher(None, s1, s2).find_longest_match(0, len(s1), 0, len(s2))
    return s1[match.a: match.a + match.size]

import numpy as np
def main():
    print(f"🔍 בונה קבוצות (Groups) מבוססות רכיבים קשירים...")
    print(f"הגדרת קשר (Edge) בין שתי דגימות:")
    print(f"1️⃣ אותו רצף miRNA.")
    print(f"2️⃣ חפיפת פרגמנטים >= {FRAGMENT_OVERLAP_THRESHOLD}nt.")
    print(f"3️⃣ ה-Site זהה לחלוטין ומוכל בתוך אזור החפיפה.\n")

    if not POS_FILE.exists():
        print(f"⚠️ קובץ המקור לא נמצא: {POS_FILE}")
        return

    df_pos = pd.read_csv(POS_FILE, low_memory=False)

    if 'site' not in df_pos.columns or 'miRNA sequence' not in df_pos.columns:
        print("⚠️ חסרות עמודות 'site' או 'miRNA sequence' בקובץ המקור!")
        return

    # 1. קיבוץ ראשוני לפי miRNA בלבד
    data_by_mirna = defaultdict(list)
    all_interactions = []

    for _, row in df_pos.iterrows():
        inter_id = str(row['ID_interaction'])
        mirna = str(row['miRNA sequence']).upper().strip()
        frag = str(row['fragment']).upper().strip()
        site = str(row['site']).upper().strip()

        if len(site) < 5 or len(frag) < 10:
            continue

        data_by_mirna[mirna].append({
            'id': inter_id,
            'fragment': frag,
            'site': site
        })
        all_interactions.append(inter_id)

    print(f"✅ נטענו {len(all_interactions)} אינטראקציות חיוביות תקינות.")
    print("-" * 60)

    # 2. בניית גרף הקשרים לפי המשפך המדויק
    adjacency = defaultdict(list)

    for mirna, instances in data_by_mirna.items():
        n = len(instances)
        for i in range(n):
            for j in range(i + 1, n):
                inst1 = instances[i]
                inst2 = instances[j]

                # שלב א': חפיפת פרגמנטים
                overlap_string = get_longest_common_substring_fast(inst1['fragment'], inst2['fragment'])

                if len(overlap_string) >= FRAGMENT_OVERLAP_THRESHOLD:
                    # שלב ב': ה-Site זהה ונופל בתוך החפיפה
                    if inst1['site'] == inst2['site'] and inst1['site'] in overlap_string:
                        id1, id2 = inst1['id'], inst2['id']
                        adjacency[id1].append(id2)
                        adjacency[id2].append(id1)

    # 3. חילוץ רכיבים קשירים (Connected Components) ליצירת Group IDs
    visited = set()
    components = []
    group_mapping = []

    group_id_counter = 1
    for inter_id in all_interactions:
        if inter_id not in visited:
            # BFS לסריקת הרכיב הקשיר (מציאת כל מי שמחובר בשרשרת)
            queue = [inter_id]
            visited.add(inter_id)
            current_component = []

            while queue:
                curr = queue.pop(0)
                current_component.append(curr)
                group_mapping.append({'ID_interaction': curr, 'Group_ID': group_id_counter})

                for neighbor in adjacency[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            components.append(current_component)
            group_id_counter += 1

    # 4. סיכום ויצוא
    multi_item_groups = [c for c in components if len(c) > 1]

    print(f"📊 תוצאות יצירת הקבוצות (Grouping):")
    print(f"    - סה\"כ דגימות שעובדו: {len(all_interactions)}")
    print(f"    - סה\"כ קבוצות ייחודיות שנוצרו (Group IDs): {len(components)}")
    print(f"    - מתוכן, קבוצות המכילות יותר מדגימה אחת (אינטראקציות חופפות פונקציונלית): {len(multi_item_groups)}")
    print("-" * 60)
    sizes = [len(c) for c in components]
    for c in components:
        if len(c)>10:
            print(len(c))
            print(c)

    print(f"Largest group: {max(sizes)}")
    print(f"Average group size: {np.mean(sizes):.2f}")
    print(f"Median group size: {np.median(sizes)}")
    # יצירת קובץ CSV עם המיפוי
    df_groups = pd.DataFrame(group_mapping)
    print(f"💾 קובץ המיפוי נשמר בהצלחה: {OUTPUT_GROUPS_FILE}")
    print(f"    מוכן לשימוש כפרמטר 'groups' ב-StratifiedGroupKFold.")


if __name__ == "__main__":
    main()


import pandas as pd
from pathlib import Path

# ===== נתיב לקובץ =====

# ===== האינטראקציות של הקבוצה =====
group_ids = [
    "darnell_human_1614",
    "darnell_human_1616",
    "darnell_human_1618",
    "darnell_human_1622",
    "darnell_human_1629",
    "darnell_human_1631",
    "darnell_human_1634",
    "darnell_human_1635",
    "darnell_human_1644",
    "darnell_human_3054",
    "darnell_human_3063",
]

# ===== קריאת הקובץ =====
df = pd.read_csv(POS_FILE, low_memory=False)

# ===== העמודות המעניינות =====
cols = [
    "ID_interaction",
    "paper name",
    "miRNA ID",
    "miRNA sequence",
    "Gene_ID",
    "site",
    "fragment",
    "sequence",
    "start",
    "end"
]

# ===== סינון =====
group_df = df[df["ID_interaction"].isin(group_ids)][cols]

print("=" * 120)
print(f"Number of interactions: {len(group_df)}")
print("=" * 120)

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", None)

print(group_df.sort_values("ID_interaction"))

# -------------------------------------------------------
# קצת סטטיסטיקה שתעזור להבין למה נוצרה הקבוצה
# -------------------------------------------------------

print("\n" + "=" * 120)
print("Summary")
print("=" * 120)

print(f"Unique miRNA IDs        : {group_df['miRNA ID'].nunique()}")
print(f"Unique miRNA sequences  : {group_df['miRNA sequence'].nunique()}")
print(f"Unique Gene IDs         : {group_df['Gene_ID'].nunique()}")
print(f"Unique sites            : {group_df['site'].nunique()}")
print(f"Unique papers           : {group_df['paper name'].nunique()}")

print("\nmiRNA IDs:")
print(group_df["miRNA ID"].value_counts())

print("\nGene IDs:")
print(group_df["Gene_ID"].value_counts())

print("\nSites:")
print(group_df["site"].value_counts())

print("\nPapers:")
print(group_df["paper name"].value_counts())


import pandas as pd
from pathlib import Path


GROUP = [
    'darnell_human_1614',
    'darnell_human_1616',
    'darnell_human_1618',
    'darnell_human_1622',
    'darnell_human_1629',
    'darnell_human_1631',
    'darnell_human_1634',
    'darnell_human_1635',
    'darnell_human_1644',
    'darnell_human_3054',
    'darnell_human_3063'
]

df = pd.read_csv(POS_FILE, low_memory=False)

df = df[df["ID_interaction"].isin(GROUP)].copy()

cols = [
    "ID_interaction",
    "Gene_ID",
    "miRNA ID",
    "miRNA sequence",
    "site",
    "fragment",
    "start",
    "end"
]

df = df[cols].sort_values("ID_interaction")

pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", None)

print(df.to_string(index=False))

print("\n========== UNIQUE SITES ==========")
print(df["site"].unique())

print("\nNumber of unique sites:", df["site"].nunique())

print("\n========== UNIQUE FRAGMENTS ==========")
print(df["fragment"].nunique())