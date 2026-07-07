from consts.global_consts import ROOT_PATH_PHD_GOAL_ONE
from utils.utilsfile import read_csv, to_csv, get_wrapper, get_subsequence_by_coordinates
import numpy as np
import os
import pandas as pd
from pathlib import Path
from BreathesDuplex.Duplex import Duplex
import re


class Probability_Matrix():
    def __init__(self, file_interactions, dataset_name):
        self.ID_interaction = str(file_interactions.stem).split(dataset_name)[1]
        if self.ID_interaction =="1818":
            print("")
        self.df_interactions = self.filter_interaction(read_csv(file_interactions))
        if len(self.df_interactions) == 0:
            self.df_interactions = None
            self.fragment = None
            self.miRNA = None
            self.dict_sub_optimal_interaction_duplex = None
            self.Z_normalization = None
            self.dict_sub_optimal_interaction_prob = None
            self.metrix_position_basepair_interactions = None
            self.metrix_position_basepair_prob = None

        else:
            self.df_interactions = self.strat_end_site_on_fragment(self.df_interactions)
            self.fragment = self.get_fragment()
            self.miRNA = self.get_miRNA()
            self.dict_sub_optimal_interaction_duplex = self.dict_duplex()
            self.Z_normalization = self.Z_normalization_calculate()
            self.dict_sub_optimal_interaction_prob = self.probability_struct_dict()
            self.metrix_position_basepair_interactions = self.initializing_matrix()
            self.metrix_position_basepair_prob = [[0 for _ in range(len(self.fragment))] for _ in range(len(self.miRNA))]

    def filter_interaction(self, df):

        df = df[(df["Seed_match_canonical_y"] == True) |(df["Seed_match_noncanonical_y"] == True)]
        df = df[df["MFE"] < 0]
        s_site = df['site'].astype(str)
        if s_site.str.contains('[A-Za-z]').any():
            df['site'] = s_site
            df["len_site"] = df["site"].apply(lambda x: len(x))
            df = df[df["len_site"].between(17, 32, inclusive="both")]

        df.reset_index(drop=True, inplace=True)
        return df

    def initializing_matrix(self):
        return [[[] for _ in range(len(self.fragment))] for _ in range(len(self.miRNA))]

    def dict_duplex(self):
        results = {}
        for index, row in self.df_interactions.iterrows():
            mrna_bulge = row["mrna_bulge"]
            mrna_inter = row["mrna_inter"]
            mir_inter = row["mir_inter"]
            mir_bulge = row["mir_bulge"]
            dp = Duplex.fromStrings(mrna_bulge, mrna_inter, mir_inter, mir_bulge)
            results[index] = {"duplex":dp,  "start_site": row["start_on_fragment"],
                              "end_site": row["end_on_fragment"]}

        return results

    def get_fragment(self):
        try:
            return self.df_interactions.loc[0]['fragment']
        except:
            print(self.ID_interaction())

    def get_miRNA(self):
        return self.df_interactions.loc[0]['miRNA_sequence']

    def Z_normalization_calculate(self):
        Z = 0
        R = 0.001987 # kcal/mol/K (bolzman)
        T = 310.15 # K
        for index, row in self.df_interactions.iterrows():
            free_energy = row['MFE'] #kcal/mol
            # Compute the normalization constant Z
            Z += np.exp(-free_energy / (R * T))
        return Z

    import numpy as np

  
    def probability_struct_dict(self):
        boltzmann_constant = 0.001987
        T = 310.15
        Z = self.Z_normalization
        # Calculate probabilities for each interaction
        results = {}
        for index, row in self.df_interactions.iterrows():
            free_energy = row['MFE']
            p_s_given_x = np.exp(-free_energy * boltzmann_constant)
            p_s_given_x = p_s_given_x / Z
            results[index] = p_s_given_x

        return results

    def strat_end_site_on_fragment(self, df):
        def calc_chimera_start(seq: str, subseq: str) -> int:
            subseq = subseq.replace("#", "")
            try:
                if seq.find(subseq) == -1:
                    return -1
                return seq.find(subseq) + 1
            except AttributeError:
                return -1
        def calc_chimera_end(start: int, site: str) -> int:
            site = site.replace("#", "")
            if start == -1:
                return -1
            return start + len(site) - 1

        df["start_on_fragment"] = df.apply(func=get_wrapper(calc_chimera_start,
                                                'fragment', 'site'), axis=1)
        df["end_on_fragment"] = df.apply(func=get_wrapper(calc_chimera_end,
                                              'start_on_fragment', 'site'), axis=1)

        return df

    def display_matrix(self):
        for row in self.metrix_position_basepair_interactions:
            print(row)

    def miRNA_fragment_match_position_matrix(self):
        for key, dp in self.dict_duplex().items():
            start_site_on_fragment = dp["start_site"]
            end_site_on_fragment = dp["end_site"]
           
            basepair_list = [mir + dp["duplex"]._mrna_inter[i] for i, mir in dp["duplex"].mir_iterator() if dp["duplex"]._mrna_bulge[i]!="#"]
           
            for i in range(len(basepair_list)):
                if ' ' in basepair_list[i]:
                    continue
                self.metrix_position_basepair_interactions[i][end_site_on_fragment - i - 1].append(key)

    def matrix_insert_prob(self):
        for i in range(len(self.metrix_position_basepair_interactions) - 1, -1, -1):
            for j in range(len(self.metrix_position_basepair_interactions[i]) - 1, -1, -1):
                value_list = self.metrix_position_basepair_interactions[i][j]
                if self.metrix_position_basepair_interactions[i][j]!= []:
                    for index in value_list:
                        self.metrix_position_basepair_prob[i][j] += self.dict_sub_optimal_interaction_prob[index]

    def get_path(self, file_path):
        file_path = file_path / (str(self.ID_interaction) + ".csv")
        return file_path

    def save_to_csv(self, file_path):
        # Convert the matrix to a DataFrame
        df = pd.DataFrame(self.metrix_position_basepair_prob)
        # Save the DataFrame to a CSV file
        to_csv(df, file_path)


def run_generate_matrices():
    main_directory = ROOT_PATH_PHD_GOAL_ONE / "Data_Breath_Duplex/"

    # Iterate over all directories and files within the main directory
    for sub_dir in main_directory.iterdir():
        count = 0
        if sub_dir.is_dir():  # Check if it is a directory
            feature_step_path = sub_dir / 'feature_step'

            # Check if the 'feature_step' folder exists
            if feature_step_path.exists() and feature_step_path.is_dir():
                print(f"Entering folder: {feature_step_path}")
                
                fragment_site_list = []

                for file in feature_step_path.iterdir():
                    print(f"Found file: {file.name}")
                    last_dir = str(re.sub(r'_\d+', '', file.name).split(".csv")[0])

                    if not "dataset" in file.name:
                        last_dir = last_dir + "_dataset"

                    file_path_source = Path(os.path.join(feature_step_path, file))

                    dataset_name = last_dir.split("_dataset")[0] + "_"
                    MP = Probability_Matrix(file_path_source, dataset_name)
                    list_not_good = []

                    if MP.df_interactions is None:
                        continue
                    file_path_target = MP.get_path(ROOT_PATH_PHD_GOAL_ONE /"Data_Graph_Matrix"/ last_dir)

                    if file_path_target.exists():
                        print("The file path are exits", file_path_source)

                    MP.miRNA_fragment_match_position_matrix()
                    MP.matrix_insert_prob()
                    fragment_site_list.append(len(str(MP.get_fragment)))
                    count +=1
                    MP.save_to_csv(file_path_target)

            else:
                print(f"Folder 'feature_step' not found in: {sub_dir}")
            print("####not good interaction####")
            print(list_not_good)


# run_generate_matrices()
