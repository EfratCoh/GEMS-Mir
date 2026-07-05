import ast
import shap
import pickle
import os
from collections import Counter
from itertools import combinations
from pathlib import Path
# from utilsClassifier import mean_std
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
from seaborn import heatmap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.svm import SVC
from xgboost import XGBClassifier
# import FeatureReader
# from FeatureReader import get_reader
# from dataset import Dataset
# from consts.global_consts import ROOT_PATH, NEGATIVE_DATA_PATH, MERGE_DATA, DATA_PATH_INTERACTIONS
# from dataset import Dataset
# from utils.utilsfile import read_csv, to_csv
from utilsfile import read_csv, to_csv
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import Classifier.FeatureReader as FeatureReader
from Classifier.FeatureReader import get_reader
from Classifier.ClfLogger import logger
from consts.global_consts import ROOT_PATH, NEGATIVE_DATA_PATH, MERGE_DATA, DATA_PATH_INTERACTIONS
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import seaborn as sns
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_recall_curve
# from sklearn.metrics import f1_score
from sklearn.metrics import auc
from matplotlib import pyplot


class NoModelFound(Exception):
    pass

def model_confuse_matrix(x_test, y_test, model, model_name,s_org,d_org,name_classifiers):

    plot_confusion_matrix(model, x_test, y_test)
    plt.title(f"{s_org}_{d_org}_confuse_matrix_plot")
    fname = ROOT_PATH / Path(f"Results/figuers/{name_classifiers}/confuse_matrix/") / f"{s_org}_{d_org}.png"
    plt.savefig(fname, bbox_inches='tight',  dpi=300)
    plt.show()
    plt.clf()



def measurement(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred).ravel()
    # cm = confusion_matrix(y_true, y_pred)

    # print(cm)
    # TP = cm[0][0]
    # FP = cm[0][1]
    # FN = cm[1][0]
    # TN = cm[1][1]
    TP = cm[3]
    FP = cm[1]
    FN = cm[2]
    TN = cm[0]
    print("TP:", TP)
    print("FP:", FP)
    print("FN:", FN)
    print("TN:", TN)
    try:
        auc = roc_auc_score(y_true, y_pred)
    except:
        auc = 0

    d = {
        # Sensitivity, hit rate, recall, or true positive rate
        "TPR": TP / (TP + FN),
        # Specificity or true negative rate
        "TNR": TN / (TN + FP),
        # Precision or positive predictive value
        "PPV": TP / (TP + FP),
        # Negative predictive value
        "NPV": TN / (TN + FN),
        # Fall out or false positive rate
        "FPR": FP / (FP + TN),
        # False negative rate
        "FNR": FN / (TP + FN),
        # False discovery rate
        "FDR": FP / (TP + FP),
        # Overall accuracy
        "ACC": (TP + TN) / (TP + FP + FN + TN),
        # roc auc
        "AUC": auc,
        "F1": f1_score(y_true, y_pred),
        "PNR": 0
    }

    return {k: round(v,3) for k, v in d.items()}


# this function response on load the clf from Result dir
def get_presaved_clf(results_dir: Path, dataset: str, method: str):
    clf_file = results_dir / f"{dataset}.model"
    if not clf_file.is_file():
        raise NoModelFound(f"No model found: {clf_file}")
    with clf_file.open("rb") as f:
        clf = pickle.load(f)
    return clf



def remove_numbers(string):
    return ''.join(char for char in string if not char.isdigit())


def clean_name(name):
    name_clean = name.replace("model:", "").replace("human", "").replace("darnell_human", "").replace("test:",
                                                                                                      "").replace(
        "ViennaDuplex", "").replace("_darnell_", "").replace("__", "_").replace("negative", "").replace("_features","")

    name_clean = name_clean.replace("_nucleotides", "_mono", )
    name_clean = name_clean.replace("denucleotides", "_di")
    name_clean = name_clean.replace("method1", "mrna")
    name_clean = name_clean.replace("method2", "site")
    name_clean = name_clean.replace("method3", "mockMirna")
    name_clean = name_clean.replace("rna", "RNA")

    if name_clean == "mockMirnadarnell":
        name_clean = "mockMirna"
    if name_clean == "mockMrnadarnell":
        name_clean = "mockMrna"
    if name_clean == "nonoverlappingsitesdarnell":
        name_clean = "Non site"
    if name_clean == "mockmra__di_mockMirna":
        name_clean = "mockMrna__di_fragment_mockMirna"
    if name_clean == "MockmRNA_di_mockmiRNA":
        name_clean = "mockMrna_di_fragment_mockMirna"
    if name_clean == "mockMrna_mono_mockMirna":
        name_clean = "mockMrna__mono_fragment_mockMirna"
    if name_clean == "MockmRNA_mono_mockmiRNA":
        name_clean = "mockMrna_mono_fragment_mockMirna"
    if name_clean == "mockMrna__di_site":
        name_clean = "mockMrna__di_fragment"
    if name_clean == "MockmRNA_di_site":
        name_clean = "mockMRNA_di_fragment"
    if name_clean == "MockmRNA_mono_site":
        name_clean = "mockMRNA_mono_fragment"
    name_clean = name_clean.capitalize()

    return name_clean

def plot_feature_importances(clf: XGBClassifier, X_train, s_org, d_org, name_classifiers):

    feat_imp = pd.DataFrame({'importance': clf.feature_importances_})
    feat_imp['feature'] = X_train.columns
    feat_imp.sort_values(by='importance', ascending=False, inplace=True)
    feat_imp = feat_imp.iloc[:10]
    s_org = clean_name(s_org)
    d_org = clean_name(d_org)
    title = f"Train: {s_org}- Test: {d_org}"
    feat_imp.sort_values(by='importance', inplace=True)
    feat_imp = feat_imp.set_index('feature', drop=True)
    feat_imp.plot.barh(title=title)
    plt.xlabel('Feature Importance Score')
    plt.gca().tick_params(axis="y", pad=150)
    plt.yticks(ha='left')

    # fname = ROOT_PATH / Path("Results/figuers/feature_importance/") / f"{s_org}_{d_org}_summary_plot.pdf"
    fname = ROOT_PATH / Path(f"Results/figuers/{name_classifiers}/feature_importance/") / f"{s_org}_{d_org}_summary_plot.png"
    # plt.savefig(fname, bbox_inches='tight',  dpi=300)
    plt.show()
    plt.clf()

def model_shap_plot(test, model, model_name,s_org,d_org,name_classifiers, dependence_feature=None):

    shap_values = shap.TreeExplainer(model).shap_values(test.values.astype('float'))
    shap.summary_plot(shap_values, test, show=False, max_display=10,color_bar=True, feature_names=test.columns)

    s_org = clean_name(s_org)
    d_org = clean_name(d_org)
    plt.title(f"Train: {s_org}-Test: {d_org}")
    fname = ROOT_PATH / Path(f"Results/figuers/{name_classifiers}/shap/") / f"{s_org}_{d_org}_summary2_plot.png"
    plt.gcf().axes[-1].set_aspect(100)
    plt.gcf().axes[-1].set_box_aspect(100)
    plt.gca().tick_params(axis="y", pad=150)
    plt.yticks(ha='left')
    plt.savefig(fname, bbox_inches='tight', dpi=300)
    plt.show()
    plt.clf()



def different_results_summary_one_class(method_split: str, model_dir: str, number_iteration: int, name_classifier: str,method_name):

    ms_table = None
    results_dir = ROOT_PATH / Path("Results")
    number_iteration = str(number_iteration)
    results_dir_models = ROOT_PATH / Path("Results/models") / model_dir / number_iteration
    test_dir = DATA_PATH_INTERACTIONS / "test" / method_split / number_iteration
    res_table: DataFrame = pd.DataFrame()
    FeatureReader.reader_selection_parameter = "without_hot_encoding"
    feature_reader = get_reader()

    clf_datasets = [f.stem.split("_"+ name_classifier)[0] for f in results_dir_models.glob("*.model")]
    method = name_classifier
    for clf_dataset in clf_datasets:
        for f_test in test_dir.glob("*test*"):
            f_stem = f_test.stem
            test_dataset = f_stem.split(".csv")[0]
            train_model = clean_name(clf_dataset).replace(method_name,"").replace("train","")[:-1]
            test_set = clean_name(test_dataset).replace(method_name,"").replace("test","")
            if train_model != test_set:
                print(train_model)
                print(test_set)
                continue

            print(f"clf: {clf_dataset} test: {test_dataset}, method: {method}")
            try:
                    print("##############################I am try#################")
                    clf = get_presaved_clf(results_dir_models, clf_dataset, method)
                    X_test, y_test = feature_reader.file_reader(test_dir/f"{test_dataset}.csv")

                    # score predict
                    test_score = accuracy_score(y_test, clf.predict(X_test))
                    res_table.loc[clf_dataset, test_dataset] = round(test_score, 3)
                    prediction = clf.predict(X_test)

                    # shap graph
                    # model_shap_plot(X_test, clf, method, clf_dataset, test_dataset,name_classifier, dependence_feature=None)
                    # # features importance graph
                    # plot_feature_importances(clf, X_test, clf_dataset, test_dataset, name_classifier)
                    # model_confuse_matrix(X_test, y_test, clf, method, clf_dataset, test_dataset,name_classifier)

                    ms = measurement(y_test, prediction)
                    # Predict scores for test set
                    # if method_name!="rf":
                    #   y_scores = clf.decision_function(X_test)
                    #   # # Calculate precision, recall and thresholds relate to negative
                    #   # y_test_2 = [-1 if i == 0 else 1 for i in y_test]
                    #   # y_test_2 = [0 if i == 1 else -1 for i in y_test_2]  # positive is 0
                    #   # y_test_2 = [1 if i == -1 else 0 for i in y_test_2]  # negative is 1
                    #   # y_test_2 = np.array(y_test_2)
                    #   y_test_2=y_test
                    #
                    #   precision, recall, thresholds = precision_recall_curve(y_true=y_test_2,
                    #                                                          probas_pred=y_scores)
                    #   # Calculate AUC for precision-recall-negative curve
                    #   prn_auc = auc(1-recall, precision)
                    #   print("PNR:", prn_auc)
                    #
                    #   ms['PNR'] = round(prn_auc, 3)
                    # else:
                    # get the probality for the zero class - negative
                    y_scores = clf.predict_proba(X_test)[:,0]

                    precision, recall, thresholds = precision_recall_curve(y_true=y_test,
                                                                           probas_pred=y_scores, pos_label=0)
                    prn_auc=auc(recall,precision)
                    print("PNR in the first way:", prn_auc)
                    ms['PNR'] = round(prn_auc, 3)


                    if ms_table is None:
                        ms_table = pd.DataFrame(columns=list(ms.keys()), dtype=object)
                    name_method = "model:" + clf_dataset.split("negative")[0] + "/" + "test:" + test_dataset.split("negative")[0]
                    ms_table.loc[name_method] = ms

            except NoModelFound:
                    pass

    res_table.sort_index(axis=0, inplace=True)
    res_table.sort_index(axis=1, inplace=True)

    print(res_table)
    to_csv(ms_table, results_dir /"results_iterations" / name_classifier /f"measurement_summary_{number_iteration}.csv")
    print("END result test")
    return ms_table



def clean_name(name):

    name_clean = name.replace("model:", "").replace("human", "").replace("darnell_human", "").replace("test:",
                                                                                                      "").replace(
        "ViennaDuplex", "").replace("_darnell_", "").replace("__", "")

    name_clean = name_clean.replace("_nucleotides", "_mono", )
    name_clean = name_clean.replace("denucleotides", "_di")
    name_clean = name_clean.replace("method1", "mrna")
    name_clean = name_clean.replace("method2", "site")
    name_clean = name_clean.replace("method3", "mockMirna")

    if name_clean == "mockMirnadarnell":
        name_clean = "mockMirna"
    if name_clean == "mockMrnadarnell":
        name_clean = "mockMrna"
    if name_clean == "nonoverlappingsitesdarnell":
        name_clean = "Non site"
    if name_clean == "mockMrna__di_mockMirna":
        name_clean = "mockMrna__di_fragment_mockMirna"
    if name_clean == "mockMrna__di_site":
        name_clean = "mockMrna__di_fragment"
    if name_clean == "mockMrna_mono_mockMirna":
        name_clean = "mockMrna__mono_fragment_mockMirna"
    name_clean = name_clean.replace("rna", "RNA")
    name_clean = name_clean.capitalize()

    return name_clean

def conver_name(name):
    if name == "Tarbase_liver":
        name = "TarBase_Liver"
    elif name == "Tarbase":
        name = "TarBase"
    elif name == "Tarbase_microarray":
        name = "TarBase_microarray"
    elif name == "Mockmirna":
        name = "Mock_miRNA"
    elif name == "Non_overlapping_sites":
        name = "NPS_CLASH_MFE"
    elif name == "Non_overlapping_sites_random":
        name = "NPS_CLASH_Random"
    elif name == "Mockmrna_mono_mrna":
        name = "Mock_mono_mRNA"
    elif name == "Mockmrna_di_mrna":
        name = "Mock_di_mRNA"
    elif name == "Mockmrna_mono_site":
        name = "Mock_mono_fragment"
    elif name == "Mockmrna_di_fragment":
        name = "Mock_di_fragment"
    elif name == "Mockmrna_di_site":
        name = "Mock_di_fragment"
    elif name == "Mockmrna_di_fragment_mockmirna":
        name = "Mock_di_fragment_&_miRNA"
    elif name == "Mockmrna_di_mockmirna":
        name = "Mock_di_fragment_&_miRNA"
    elif name == "Mockmrna_mono_fragment_mockmirna":
        name = "Mock_mono_fragment_&_miRNA"
    elif name == "Clip_interaction_clip_3_":
         name = "CLIP_non_CLASH"
    elif name == "Clip_interaction_clip_3":
        name = "CLIP_non_CLASH"

    elif name == "Non_overlapping_sites_cip_data":
         name = "NPS_CLIP_MFE"
    elif name == "Non_overlapping_sites_clip_data_random":
        name = "NPS_CLIP_Random"
    return name

def creat_final_figure():
    chossing_methods = ['tarBase_microArray',
                        "non_overlapping_sites", "non_overlapping_sites_random",
                        "mockMiRNA",
                        "mockMRNA_di_mRNA", "mockMRNA_di_fragment", "mockMRNA_di_fragment_mockMiRNA",
                        "clip_interaction_clip_3_",
                        "non_overlapping_sites_clip_data_random"]
    for i in range(len(chossing_methods)):
        name = chossing_methods[i]
        chossing_methods[i] = name.capitalize()
    result_dir = Path('/sise/home/efrco/efrco-master/Results/measurments_cross_validation/')
    list_classifiers= ['svm', 'isolation_forest', 'binary_comper_SVM', 'binary_comper_rf']
    measurement_name_list = ['ACC','AUC', 'TPR', 'TNR', 'PPV', 'FPR']


    res_table = pd.DataFrame()
    for classifier in list_classifiers:
            out_dir_mean = result_dir/classifier / "final_mean.csv"
            df_mean = read_csv(out_dir_mean)
            count = -1
            for measurement_name in measurement_name_list:
                count = count + 1

                for (columnName, columnData) in df_mean.iteritems():
                    train_name = clean_name(columnName.split('/')[0])
                    test_name = clean_name(columnName.split('/')[1])
                    train_name = train_name.replace("__", "_")
                    test_name = test_name.replace("__", "_")

                    if train_name=='Mockmrna_di_fragment_mockmirna':
                        print("f")

                    if train_name != test_name:
                        print(train_name)
                        continue
                    if train_name not in chossing_methods:
                        print("#################################")
                        print(train_name)
                        continue
                    row = pd.Series()
                    row['classifier'] = classifier
                    row['dataset'] = train_name
                    row['measurement_name'] = measurement_name
                    row['value'] = df_mean.loc[measurement_name, columnName]
                    res_table = res_table.append(row, ignore_index=True)

    res_table['value'] = pd.to_numeric(res_table['value'])
    df = res_table
    g = sns.catplot(data=df, x='measurement_name', y='value', hue='classifier', col='dataset',kind='bar', height=4,
                    aspect=1)

    # Set axis labels and title
    g.set_axis_labels('Measurement Name', 'Value')
    # g.fig.suptitle('Value by Measurement Name and Classifier')
    # Add legend
    # g.add_legend()

    fname = ROOT_PATH / Path(f"figuers") / "one_class_figure.png"
    plt.savefig(fname, format="PNG", bbox_inches='tight', dpi=300)

    # Show the plot
    plt.show()

# creat_final_figure()


def creat_final_figure_one_dataset(name_dataset, name_method):
    chossing_methods = ['tarBase_microArray',
                        "non_overlapping_sites", "non_overlapping_sites_random",
                        "mockMiRNA",
                        "mockMRNA_di_mRNA", "mockMRNA_di_fragment", "mockMRNA_di_fragment_mockMiRNA",

                        "clip_interaction_clip_3_",
                        "non_overlapping_sites_clip_data_random"]

    for i in range(len(chossing_methods)):
        name = chossing_methods[i]
        chossing_methods[i] = name.capitalize()
    result_dir = Path('/sise/home/efrco/efrco-master/Results/measurments_cross_validation/')
    list_classifiers= ['svm',  'binary_comper_SVM', 'isolation_forest', 'binary_comper_rf']
    conver_dict = {'svm':'One-class SVM','binary_comper_SVM': 'SVM', 'isolation_forest':'Isolation Forest',
                   'binary_comper_rf': 'Random Forest'}
    measurement_name_list = ['ACC','AUC', 'TPR', 'TNR', 'FPR']


    res_table = pd.DataFrame()
    for classifier in list_classifiers:
            out_dir_mean = result_dir/classifier / "final_mean.csv"
            df_mean = read_csv(out_dir_mean)
            count = -1
            for measurement_name in measurement_name_list:
                count = count + 1

                for (columnName, columnData) in df_mean.iteritems():
                    c = columnName
                    train_name = clean_name(columnName.split('/')[0])
                    test_name = clean_name(columnName.split('/')[1])
                    train_name = train_name.replace("__", "_")
                    test_name = test_name.replace("__", "_")

                    if train_name=='Mockmrna_di_site':
                        print("f")

                    if 'clip' in train_name and 'clip' in test_name :
                        print("f")
                    if train_name != name_dataset:
                        continue

                    if train_name != test_name:
                        print(train_name)
                        continue
                    if train_name not in chossing_methods:
                        print("#################################")
                        print(train_name)
                        continue
                    row = pd.Series()
                    row['Classifier'] = conver_dict[classifier]
                    row['dataset'] = conver_name(train_name)
                    row['Measurement'] = measurement_name
                    row['Value'] = df_mean.loc[measurement_name, columnName]
                    res_table = res_table.append(row, ignore_index=True)

    res_table['Value'] = pd.to_numeric(res_table['Value'])
    df = res_table
    palette = [sns.color_palette("rocket")[2], sns.color_palette("rocket")[4]]
    palette = [sns.color_palette("rocket")[1], sns.color_palette("rocket")[2],
               sns.color_palette("rocket")[3], sns.color_palette("rocket")[4]]
    palette =[sns.color_palette("Paired")[4],sns.color_palette("Paired")[5],sns.color_palette("Paired")[8],
              sns.color_palette("Paired")[9]]
    palette = [sns.color_palette("Paired")[4], sns.color_palette("Paired")[5], sns.color_palette("Paired")[2],
               sns.color_palette("Paired")[3]]

    sns.barplot(data=df, x='Measurement', y='Value', hue='Classifier',
                     palette= palette)

    # Set axis labels and title
    # g.set_axis_labels('Measurement Name', 'Value')
    # g.fig.suptitle('Value by Measurement Name and Classifier')
    plt.title(conver_name(name_dataset))
    # Add legend
    # g.add_legend()
    plt.legend(loc='upper left', bbox_to_anchor=(1, 0.5))

    fname = ROOT_PATH / Path(f"figuers/one_class") / f"{name_dataset}.png"
    plt.savefig(fname, format="PNG", bbox_inches='tight', dpi=300)

    # Show the plot
    plt.show()
    plt.clf()



def run():
    chossing_methods = ['tarBase_microArray',
                        "non_overlapping_sites", "non_overlapping_sites_random",
                        "mockMiRNA",
                        "mockMRNA_di_mRNA", "mockMRNA_di_site", "mockMRNA_di_fragment_mockMiRNA",
                        "clip_interaction_clip_3_",
                        "non_overlapping_sites_clip_data_random"]
    # chossing_methods = ["non_overlapping_sites_clip_data_random"]
    for i in range(len(chossing_methods)):
        name = chossing_methods[i]
        chossing_methods[i] = name.capitalize()
        #one class svm
        name_dataset= chossing_methods[i]
        creat_final_figure_one_dataset(name_dataset, name_method ='svm')

#
# run()