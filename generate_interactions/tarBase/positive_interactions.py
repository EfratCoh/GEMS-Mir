import sys
from typing import Tuple
import pandas as pd
from numpy import int64
from pandas import DataFrame
from pathlib import Path
from consts.global_consts import MERGE_DATA, ROOT_PATH, BIOMART_PATH, POSITIVE_PATH_FEATUERS
from consts.mirna_utils import MIRBASE_FILE
from utils.logger import logger
from utils.utilsfile import get_subsequence_by_coordinates, get_wrapper, to_csv
import numpy as np

# necessary_columns = set(['GI_ID', 'microRNA_name', 'miRNA sequence', 'target sequence', 'number of reads'])


def read() -> DataFrame:
    # Consts
    file_name = ROOT_PATH / "generate_interactions/tarBase/Homo_sapiens(v.9).tsv"
    usecols = ['mirna_name', 'gene_name',
               'gene_id', 'gene_location', 'transcript_name', 'transcript_id', 'microt_score']


    logger.info(f"Reading file {file_name}")
    # df: DataFrame = pd.read_excel(file_name, usecols=usecols, engine='openpyxl')
    df = pd.read_csv(file_name, usecols=usecols,  sep='\t')
    print("Number of interactions before:", df.shape[0])

    # # filter only liver interaction that will be match to darnell
    df = df[df['gene_location'] == '3UTR']
    df = df.drop_duplicates(subset=['mirna_name', 'gene_name'])
    print("Number of interactions is after gene locations :", df.shape[0])

    # drop interactions that not direct interactions

    file_name = ROOT_PATH / "generate_interactions/tarBase/interaction_mirna_up_down.csv"
    usecols = ['gene_name', 'gene_id', 'mirna_name', 'micro_tscore']

    df_tarBase_mirna_up_down = pd.read_csv(file_name, usecols=usecols)
    print("Number of interactions is in tar:", df_tarBase_mirna_up_down.shape[0])

    shared_rows = pd.merge(df_tarBase_mirna_up_down, df[['mirna_name', 'gene_name', 'transcript_id', 'transcript_name']],
                         on=['mirna_name', 'gene_name'], how='left')

    print("Number of interactions after share:", shared_rows.shape[0])
    # 37 miRNA
    print("Number of interactions after share:", shared_rows['mirna_name'].nunique())
    print(shared_rows['mirna_name'].unique())

    return shared_rows

def read_indirect() -> DataFrame:
    # Consts
    file_name = ROOT_PATH / "generate_interactions/tarBase/Homo_sapiens(v.9).tsv"
    usecols = ['mirna_name', 'gene_name',
               'gene_id', 'gene_location', 'transcript_name', 'transcript_id', 'microt_score']


    logger.info(f"Reading file {file_name}")
    # df: DataFrame = pd.read_excel(file_name, usecols=usecols, engine='openpyxl')
    df = pd.read_csv(file_name,usecols=usecols,  sep='\t')
    print("Number of interactions before:", df.shape[0])

    # # filter only liver interaction that will be match to darnell
    df = df[df['gene_location'] == '3UTR']

    df = df.drop_duplicates(subset=['mirna_name', 'gene_name'])
    print("Number of interactions is after gene locations :", df.shape[0])

    return df

def change_columns_names(df: DataFrame) -> DataFrame:
    return df.rename(columns={"mirna_name": "miRNA ID", "transcript_id": "Gene_ENST", "gene_name": "geneName","gene_id":"Gene_ID"})


def add_meta_data(df: DataFrame) -> DataFrame:
    paper_name = "tarBase"
    df.insert(0, "paper name", paper_name)
    df.insert(0, "organism", "Human")
    df["key"] = df.reset_index().index

    return df


def save(df: DataFrame, file_name: str):
    full_path = ROOT_PATH / file_name
    to_csv(df, full_path)


# In this step we remove all the interactions that exsits in clash interaction

def filter_clash_interaction(df: DataFrame):

    files_name = ["human_mapping_ViennaDuplex_features.csv", "darnell_human_ViennaDuplex_features.csv","unambiguous_human_ViennaDuplex_features.csv", "qclash_melanoma_human_ViennaDuplex_features.csv"]

    print("################number of rows before filter clash interactins:#####", df.shape[0])
    for file_name in files_name:
        file_name_path = POSITIVE_PATH_FEATUERS / file_name
        usecols = ['miRNA ID', 'Gene_ID']
        logger.info(f"Reading file {file_name_path}")
        df_positive_interaction = pd.read_csv(file_name_path, usecols=usecols)

        # transform the format of geneName
        df_positive_interaction['Gene_ID'] = df_positive_interaction['Gene_ID'].apply(lambda x: x.split("|")[0])

        # Intersection to find negative interaction wiche exists in clash poitive interacitons
        intersected_df = pd.merge(df, df_positive_interaction, how='inner', on=['miRNA ID', 'Gene_ID'])

        # remove interactions that exists in both of the dataset
        print("number of rows to remove:" + str(file_name) + " " + str(intersected_df.shape[0]))
        print("number of rows before:" + str(df.shape[0]))
        new = df[(~df.key.isin(intersected_df.key))]
        print("number of rows after:" + str(new.shape[0]))
        df = new

    print("################number of rows after filter clash interactions:##########", df.shape[0])

    return df

def drop_duplicate(data: object) -> object:

    rows = []
    group_by_duplex = data.groupby(['geneName', 'miRNA ID'])# create an empty DataFrame to store the randomly selected rows
    selected_rows = pd.DataFrame()

    # loop over each group and select a random row from each group
    for name, group in group_by_duplex:
        selected_rows = selected_rows.append(group.iloc[np.random.randint(0, len(group))])

    return selected_rows


def mrna_sequences(df: DataFrame):
    size_befor = df.shape[0]
    file_name = BIOMART_PATH / "human_3utr.csv"
    logger.info(f"Reading file {file_name}")
    df_human_3utr = pd.read_csv(file_name)
    df_human_3utr['Gene_ID'] = df_human_3utr['ID'].apply(lambda x: x.split("|")[0])

    # In cases where multiple UTRs exist per gene, we considered the
    # longest UTR.
    df_human_3utr = df_human_3utr.loc[df_human_3utr.groupby(['Gene_ID'])["sequence length"].idxmax()]
    df_human_3utr = df_human_3utr.rename(columns={'sequence': 'full_mrna'})

    ##############
    # mrna_seq_list = df_human_3utr["Gene_ID"].tolist()
    # df['new'] = df['Gene_ID'].apply(lambda x: x not in mrna_seq_list)
    # # print(df['new'] == True)

    # for each mRNA we find the longest sequence of the target
    df = pd.merge(df, df_human_3utr, how='inner', on=['Gene_ID'])

    # clean mrna that samll
    df = df[df['full_mrna'].apply(lambda x: len(x)) > 40]

    df = df.drop(columns=['Gene_ID', 'Unnamed: 0', 'sequence length'], axis=1)
    df = df.rename(columns={"ID": "Gene_ID", 'sequence': 'full_mrna'})

    # all the interactions that was found for them the mrna sequence is 3UTR
    df["region"] ='3utr'
    return df


def mirna_sequences(df: DataFrame):

    file_name = MIRBASE_FILE
    logger.info(f"Reading file {file_name}")
    df_mirna = pd.read_csv(file_name)
    df_mirna_hsa = df_mirna[df_mirna['miRNA ID'].str.contains('hsa')]
    # df_mirna_hsa = df_mirna_hsa.rename(columns={"miRNA ID": "microRNA_name"})

    # In cases where multiple mirna exist because the different release , we considered the
    # last release.
    df_mirna_hsa['version'] = pd.to_numeric(df_mirna_hsa['version'])

    df_mirna_hsa = df_mirna_hsa.loc[df_mirna_hsa.groupby(['miRNA ID'])["version"].idxmax()]

    # for each mirna we find the longest sequence of the mirna
    df = pd.merge(df, df_mirna_hsa, how='inner', on=['miRNA ID'])
    df = df.drop(labels=['version', 'prefix', 'Unnamed: 0'], axis=1)

    return df


def run(out_filename):

    # df = read()
    df = read_indirect()

    df = change_columns_names(df)
    df = add_meta_data(df)
    # step-1
    # df = filter_clash_interaction(df)
    # # step-2
    # print("before mrna:", df.shape[0])
    #
    # df = mrna_sequences(df)
    # print("before mirna:", df.shape[0])
    # # step-3
    # df = mirna_sequences(df)
    # print("after:", df.shape[0])

    # reorder to columns
    column_names = ['index', 'Source', 'organism', 'miRNA ID',
                    'Gene_ID', 'miRNA sequence', 'full_mrna', 'method', 'tissue']
    logger.info("replace T with U")
    # seq_cols = ['miRNA sequence', 'full_mrna']
    # df[seq_cols] = df[seq_cols].replace(to_replace='T', value='U', regex=True)

    # df = df.reindex(columns=column_names)
    # df = drop_duplicate(df)

    # step-4
    print("final interactions in tarBase reade:", df.shape[0])
    save(df, out_filename)


if __name__ == '__main__':
    run("generate_interactions/tarBase/tarBase_human_positive_V9.0.csv")

    path = ROOT_PATH / "generate_interactions/tarBase/tarBase_human_negative.csv"
    # df = pd.read_csv(Path(path))
    # print(df)
