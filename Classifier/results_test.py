# from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
# from sklearn.metrics import f1_score
# from sklearn.metrics import confusion_matrix
# import Classifier.FeatureReader as FeatureReader
# from Classifier.FeatureReader import get_reader
# from consts.global_consts import ROOT_PATH_PHD_GOAL_ONE,DATA_TRAIN_TEST, NEGATIVE_DATA_PATH, MERGE_DATA, DATA_PATH_INTERACTIONS
# import pandas as pd
# import pickle
# from sklearn.metrics import accuracy_score
#
#
# def results_summary(exp_id, model_mode, reader_mode,base_dir):
#
#     results_dir_models = base_dir / model_mode  #folder of the model
#     csv_base_dir = base_dir / "Data_merge_embedding_features" #folder of the data test
#     summary_results = []
#
#     FeatureReader.reader_selection_parameter = reader_mode
#     feature_reader = get_reader()
#     feature_reader.setnumbrtexperiments(exp_id)
#
#     # Loop over fold directories
#     for fold_path in sorted(results_dir_models.glob("fold*")):
#         if not fold_path.is_dir():
#             continue
#
#         number_fold = fold_path.name.replace("fold", "")
#
#         for model_file in sorted(fold_path.glob("*.model")):
#             model_name = model_file.stem
#             try:
#                 parts = model_name.split("_")
#                 number_epoch = parts[3]
#             except IndexError:
#                 print(f"Skipping invalid model name: {model_name}")
#                 continue
#
#             try:
#                 with model_file.open("rb") as f:
#                     model = pickle.load(f)
#             except Exception as e:
#                 print(f"Failed to load model {model_file}: {e}")
#                 continue
#
#             test_dir = csv_base_dir / f"fold{number_fold}" / f"epoch_{number_epoch}"
#             test_dataset = f"merged_test_fold_{number_fold}_epoch_{number_epoch}"
#             test_csv_path = test_dir / f"{test_dataset}.csv"
#
#             if not test_csv_path.exists():
#                 print(f"Test file not found: {test_csv_path}")
#                 continue
#
#             try:
#                 X_test, y_test = feature_reader.file_reader(test_csv_path)
#             except Exception as e:
#                 print(f"Error reading test set {test_csv_path}: {e}")
#                 continue
#
#             try:
#                 y_pred = model.predict(X_test)
#                 y_prob = model.predict_proba(X_test)[:, 1]
#
#                 acc = accuracy_score(y_test, y_pred)
#                 f1 = f1_score(y_test, y_pred)
#                 auc = roc_auc_score(y_test, y_prob)
#                 print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
#                 print("fold:", number_fold)
#                 print("epoch:" , number_epoch)
#                 print("ACC:", acc)
#
#                 tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
#
#             except Exception as e:
#                 print(f"Error during evaluation for {model_file}: {e}")
#                 continue
#
#             summary_results.append({
#                 "fold": number_fold,
#                 "epoch": number_epoch,
#                 "model_file": model_file.name,
#                 "test_file": test_csv_path.name,
#                 "accuracy": acc,
#                 "f1_score": f1,
#                 "auc": auc,
#                 "tp": tp,
#                 "tn": tn,
#                 "fp": fp,
#                 "fn": fn
#             })
#
#             print(f"Evaluated model {model_file.name} — Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
#
#     # Save summary
#     results_df = pd.DataFrame(summary_results)
#     output_file = base_dir / "Results" / f"evaluation_summary_{model_mode}.csv"
#     results_df.to_csv(output_file, index=False)
#     print(f"Saved full evaluation summary to: {output_file}")
#
#
# # results_summary(model_mode="models_xgbs_combine_embedding", reader_mode="with_embedding_vector" )
#
# # results_summary(model_mode="models_xgbs_without_embedding", reader_mode="without_embedding_vector" )
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, f1_score, average_precision_score
import Classifier.FeatureReader as FeatureReader
from Classifier.FeatureReader import get_reader
from consts.global_consts import ROOT_PATH_PHD_GOAL_ONE, DATA_TRAIN_TEST, NEGATIVE_DATA_PATH, MERGE_DATA, DATA_PATH_INTERACTIONS
import pandas as pd
import numpy as np
import pickle
import os
import shap # <-- חובה לייבא את SHAP

def results_summary(exp_id, model_mode, reader_mode, base_dir):
    results_dir_models = base_dir / model_mode
    csv_base_dir = base_dir / "Data_merge_embedding_features"

    results_out_dir = base_dir / "Results"
    os.makedirs(results_out_dir, exist_ok=True)

    summary_results = []
    predictions_results = []
    shap_importances = []  # <-- שינינו את השם שיהיה ברור שאלו ערכי SHAP

    FeatureReader.reader_selection_parameter = reader_mode
    feature_reader = get_reader()
    feature_reader.setnumbrtexperiments(exp_id)

    # Loop over fold directories
    for fold_path in sorted(results_dir_models.glob("fold*")):
        if not fold_path.is_dir():
            continue

        number_fold = fold_path.name.replace("fold", "")

        for model_file in sorted(fold_path.glob("*.model")):
            model_name = model_file.stem
            try:
                parts = model_name.split("_")
                number_epoch = parts[3]
            except IndexError:
                print(f"Skipping invalid model name: {model_name}")
                continue

            try:
                with model_file.open("rb") as f:
                    model = pickle.load(f)
            except Exception as e:
                print(f"Failed to load model {model_file}: {e}")
                continue

            test_dir = csv_base_dir / f"fold{number_fold}" / f"epoch_{number_epoch}"
            test_dataset = f"merged_test_fold_{number_fold}_epoch_{number_epoch}"
            test_csv_path = test_dir / f"{test_dataset}.csv"

            if not test_csv_path.exists():
                print(f"Test file not found: {test_csv_path}")
                continue

            try:
                X_test, y_test = feature_reader.file_reader(test_csv_path)
            except Exception as e:
                print(f"Error reading test set {test_csv_path}: {e}")
                continue

            try:
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]

                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_prob)
                auprc = average_precision_score(y_test, y_prob)

                tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

                print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
                print("fold:", number_fold, "| epoch:", number_epoch)
                print(f"ACC: {acc:.4f} | AUC: {auc:.4f} | F1: {f1:.4f}")

                for true_label, prob in zip(y_test, y_prob):
                    predictions_results.append({
                        "fold": number_fold,
                        "epoch": number_epoch,
                        "True_Label": true_label,
                        "Prob_Score": prob
                    })

                # <--- 2. חישוב SHAP במקום Feature Importance רגיל --->
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_test)

                    # XGBoost מחזיר לפעמים רשימה של 2 מערכים לבעיות בינאריות (אחד לכל מחלקה)
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1]

                    # חישוב הערך המוחלט הממוצע לכל פיצ'ר (Global SHAP Importance)
                    mean_abs_shap = np.abs(shap_values).mean(axis=0)

                    feat_names = X_test.columns if isinstance(X_test, pd.DataFrame) else [f"Feature_{i}" for i in range(X_test.shape[1])]

                    for name, mean_shap in zip(feat_names, mean_abs_shap):
                        shap_importances.append({
                            "fold": number_fold,
                            "epoch": number_epoch,
                            "feature": name,
                            "mean_absolute_shap": mean_shap # <-- נשמר כמדד הוגן ומדויק
                        })
                except Exception as shap_e:
                    print(f"Error calculating SHAP for {model_file.name}: {shap_e}")

            except Exception as e:
                print(f"Error during evaluation for {model_file}: {e}")
                continue

            summary_results.append({
                "fold": number_fold,
                "epoch": number_epoch,
                "model_file": model_file.name,
                "test_file": test_csv_path.name,
                "accuracy": acc,
                "f1_score": f1,
                "auc": auc,
                "auprc": auprc,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn
            })

    metrics_file = results_out_dir / f"gems_metrics_{model_mode}.csv"
    pd.DataFrame(summary_results).to_csv(metrics_file, index=False)

    preds_file = results_out_dir / f"gems_predictions_{model_mode}.csv"
    pd.DataFrame(predictions_results).to_csv(preds_file, index=False)

    # שמירת קובץ ה-SHAP החדש
    feats_file = results_out_dir / f"gems_shap_importance_{model_mode}.csv"
    if shap_importances:
        pd.DataFrame(shap_importances).to_csv(feats_file, index=False)

    print("\n✅ Evaluation Complete!")
    print(f"1. Metrics saved to: {metrics_file}")
    print(f"2. Predictions saved to: {preds_file}")
    if shap_importances:
        print(f"3. SHAP Importances saved to: {feats_file}")
# דוגמאות קריאה לפונקציה
# results_summary(model_mode="models_xgbs_combine_embedding", reader_mode="with_embedding_vector", base_dir=ROOT_PATH_PHD_GOAL_ONE)
# results_summary(model_mode="models_xgbs_without_embedding", reader_mode="without_embedding_vector", base_dir=ROOT_PATH_PHD_GOAL_ONE)