from utils.utilsfile import read_csv, to_csv
from consts.global_consts import ROOT_PATH_PHD_GOAL_ONE
from Prepare_Data.rna_site_insertion_negative import get_site_from_extended_site
from Prepare_Data.normalization_final_step_negative import finalize
from Prepare_Data.feature_extraction import feature_extraction
from consts.global_consts import MERGE_DATA
from Prepare_Data.Duplex_step import duplex as duplex_positive
import os
import pandas as pd
from pathlib import Path
import re
from consts.datasets_path import POSITIVE_DATASETS_ARR, NEGATIVE_DATASETS_ARR, neg_dataset_NPS_CLASH

def full_pipline(dataset_file, path_dir_target):

    # step 1- extract duplex of the interaction by VieannaDuplex
    print("###############Duplex#############")
    path_dir_target_duplex = path_dir_target / "duplex_step"
    duplex_positive('ViennaDuplex', dataset_file, path_dir_target_duplex)

    # For each MTI we calculate features
    for fin in path_dir_target_duplex.glob("*.csv"):

        file_name = os.path.basename(fin)

        path_dir_target_feature_step = path_dir_target / "feature_step" / file_name
        if path_dir_target_feature_step.exists():
            print("The directory interaction exists, proceeding...", path_dir_target_feature_step)
            continue

        # step 2- extract the site and its coordinates
        print("###############Site#############")

        interaction_after_duplex = read_csv(fin)
        interaction_after_site = get_site_from_extended_site(interaction_after_duplex)

        print("###############Normaliztion#############")

        # step 3- normalization of the dataframe
        interaction_after_normalization = finalize(interaction_after_site)

        # step 4 - extract features
        print("###############extract features#############")

        interaction_after_features = feature_extraction(interaction_after_normalization)
        interaction_after_features.rename(columns={'fragment_x': 'fragment', "sequence":"mRNA_sequence"}, inplace=True)
        interaction_after_features.drop(columns=['fragment_y'], inplace=True)
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        file_name = os.path.basename(fin)

        path_dir_target_feature_step = path_dir_target / "feature_step" / file_name
        to_csv(interaction_after_features, path_dir_target_feature_step)
        
###############################################################################################################################################################
def not_extended_site():

    from Bio.Seq import Seq
    from Bio.SeqFeature import FeatureLocation
    from utils.utilsfile import get_subsequence_by_coordinates, get_wrapper, read_csv, to_csv


    def get_subsequence_by_coordinates(full_sequence: str, start: int, end: int, strand="+",
                                       extra_chars: int = 3) -> str:
        start = int(start)
        end = int(end)

        assert strand=='+' or strand=='-', "strand value incorrect {}".format(strand)
        strand = 1 if strand == '+' else -1

        start = max(0, start + extra_chars)  # because python is zero based
        end = min(len(full_sequence), end - extra_chars)

        if (end - start) <= 0:
            raise ValueError(f"Error:no subsequence to extract. start={start}, end={end}, full_seq_len={len(full_sequence)}")
        seq = Seq(full_sequence)
        feature_loc: FeatureLocation = FeatureLocation(start, end, strand=strand)
        sub_sequence: Seq = feature_loc.extract(seq)

        # print("utils:", len(sub_sequence))
        #
        # print("utils:", sub_sequence)

        return str(sub_sequence)


    def get_subsequence_by_coordinates2(full_sequence: str, start: int, end: int, strand="+",
                                       extra_chars: int = 0) -> str:
        start = int(start)
        end = int(end)

        assert strand=='+' or strand=='-', "strand value incorrect {}".format(strand)
        strand = 1 if strand == '+' else -1

        start = max(0, start - 1 - extra_chars)  # because python is zero based
        end = min(len(full_sequence), end + extra_chars)

        if (end - start) <= 0:
            raise ValueError(f"Error:no subsequence to extract. start={start}, end={end}, full_seq_len={len(full_sequence)}")
        seq = Seq(full_sequence)
        feature_loc: FeatureLocation = FeatureLocation(start, end, strand=strand)
        sub_sequence: Seq = feature_loc.extract(seq)

        # print("utils:", len(sub_sequence))
        #
        # print("utils:", sub_sequence)

        return str(sub_sequence)


    def calc_chimera_start(seq: str, subseq: str) -> int:
        subseq = subseq.replace("#", "")
        try:
            if seq.find(subseq) == -1:
                print(seq)
                print(subseq)
                return -1
            return seq.find(subseq) + 1
        except AttributeError:
            return -1

    def calc_chimera_end(start: int, site: str) -> int:
        site = site.replace("#", "")
        if start == -1:
            return -1
        return start + len(site) - 1


    pos_dir_name = MERGE_DATA / "positive_interactions_new/data_without_featuers/darnell_human_ViennaDuplex.csv"
    df = read_csv(pos_dir_name)
    df["start"] = df.apply(func=get_wrapper(calc_chimera_start,
                                            'full_mrna', 'site'), axis=1)
    df["end"] = df.apply(func=get_wrapper(calc_chimera_end,
                                          'start', 'site'), axis=1)
  
    df["site"] = df.apply(func=get_wrapper(get_subsequence_by_coordinates,
                                           "full_mrna", "start", "end", extra_chars=3),axis=1)
    df["start"] = df.apply(func=get_wrapper(calc_chimera_start,
                                            'full_mrna', 'site'), axis=1)
    df["end"] = df.apply(func=get_wrapper(calc_chimera_end,
                                          'start', 'site'), axis=1)
    df["len"] = df["site"].apply(lambda x: len(x))
    df.drop(columns=['len', 'start', 'end'], inplace=True)
    target_name = MERGE_DATA / "positive_interactions_new/data_without_featuers/darnell_human_ViennaDuplex_75nt_fragment.csv"
    to_csv(df,target_name)
    
###############################################################################################################################################################
def generate_positive_interaction():
    pos_dir_name = MERGE_DATA / "positive_interactions_new/data_without_featuers/"
    for dataset_file in pos_dir_name.glob("*.csv"):
        dataset_name = str(dataset_file.stem).split("_features.csv")[0].split("_features")[0].split("_ViennaDuplex")[0]
        print(dataset_name)
        df = read_csv(dataset_file)
        df['ID_interaction'] = [f'{dataset_name}_{i}' for i in range(0, len(df))]
        data_set_dir_name = dataset_name + "_dataset"
        to_csv(df, dataset_file)
        path = ROOT_PATH_PHD_GOAL_ONE / "Data_Breath_Duplex/"
        path_dir_target = path / data_set_dir_name
        full_pipline(dataset_file, path_dir_target)
        break

###############################################################################################################################################################
def generate_negative_interaction():
    dataset_file = neg_dataset_NPS_CLASH
    df = read_csv(dataset_file)
    data_set_dir = "NPS_darnell_human"
    df['ID_interaction'] = [f'{data_set_dir}_{i}' for i in range(0, len(df))]
    data_set_dir_name = data_set_dir + "_dataset"
    to_csv(df, dataset_file)
    path = ROOT_PATH_PHD_GOAL_ONE / "Data_Breath_Duplex/"
    path_dir_target = path / data_set_dir_name
    full_pipline(dataset_file, path_dir_target)


###############################################################################################################################################################
