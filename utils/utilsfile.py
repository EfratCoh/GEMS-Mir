import datetime
from multiprocessing import Pool, Process
from pathlib import Path
from typing import Callable, Tuple
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import FeatureLocation
import pandas as pd
import numpy as np
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from pandas import DataFrame
from Bio.Blast.Applications import NcbiblastnCommandline
from consts.biomart import BIOMART_DATA_PATH
# import os
# os.environ['PATH'] = "/sise/home/efrco/efrco-master/utils"
import sys
import os
import shutil
sys.path.append('/home/efrco/efrco-master/utils/')
from consts.global_consts import ROOT_PATH_PHD_GOAL_ONE
import importlib
import importlib.util

from logger import logger
# from utils.logger import logger
import subprocess
from more_itertools import chunked


def filename_suffix_append(f, s):
        f = Path(f)
        return str(f.parent / Path(f.stem + s + f.suffix))

def filename_date_append(f):
        return str(filename_suffix_append(f, "_{}".format(datetime.now().strftime("%Y%m%d-%H%M%S"))))

def drop_unnamed_col(df):
    for c in df.columns:
        if c.find ("Unnamed")!=-1:
            df.drop([c], axis=1, inplace=True)

def get_wrapper(func, *columns, **kwargs):
    def wrapper(row):
        return func(*[row[c] for c in columns], **kwargs)
    return wrapper

#
# def parallelize_apply(df, func, n_cores=4):
#     df_split = np.array_split(df, n_cores)
#     pool = Pool(n_cores)
#     df = pd.concat(pool.map(func, df_split))
#     pool.close()
#     pool.join()
#     return df


def get_subsequence_by_coordinates(full_sequence: str, start: int, end: int, strand="+",
                                   extra_chars: int = 0) -> str:
    if start == -1:
        raise ValueError(f"Error:no subsequence to extract. start={start}, end={end}")

    start = int(start)
    end = int(end)

    assert strand=='+' or strand=='-', "strand value incorrect {}".format(strand)
    strand = 1 if strand == '+' else -1
    start = max(0, start - 1 - extra_chars) # because python is zero based
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



def get_subsequence_by_coordinates_no_exception(full_sequence: str, start: int, end: int, strand="+",
                                   extra_chars: int = 0) -> str:
    try:
        return get_subsequence_by_coordinates(full_sequence, start, end, strand, extra_chars)
    except ValueError as e:
        return str(e)

def to_csv(df, path: Path) -> None:
    df.reset_index(inplace=True)
    df.loc[-1] = df.dtypes
    df.index = df.index + 1
    df.sort_index(inplace=True)
    df.to_csv(path, index=False)
    logger.info(f"Saved file {path}")


def read_csv(path: Path) -> DataFrame:
   try:
        logger.info(f"read file {str(path)}")
        # Read types first line of csv
        dtypes = pd.read_csv(path, nrows=1).iloc[0].to_dict()
        # Read the rest of the lines with the types from above
        d = pd.read_csv(path, dtype=dtypes, skiprows=[1], index_col=0)
        logger.info(f"read shape: {d.shape}")
   except:
       logger.info((f"read file without datatypes"))
       d = pd.read_csv(path)

   return d




# def read_csv(path: Path) -> DataFrame:
#    try:
#        try:
#            logger.info(f"read file {str(path)}")
#
#            # Read first data row
#            first_row = pd.read_csv(path, nrows=1)
#
#            # Convert first row to dict
#            dtypes_candidate = first_row.iloc[0].to_dict()
#
#            # Define valid pandas dtypes (extend if needed)
#            valid_dtypes = {
#                "int", "int64",
#                "float", "float64",
#                "str", "string",
#                "bool", "object"
#            }
#
#            # Check if ALL values in the row look like dtypes
#            is_dtype_row = all(
#                isinstance(v, str) and v.lower() in valid_dtypes
#                for v in dtypes_candidate.values()
#            )
#
#            if is_dtype_row:
#                logger.info("Detected dtype definition row — using it.")
#                d = pd.read_csv(
#                    path,
#                    dtype=dtypes_candidate,
#                    skiprows=[1],
#                    index_col=0
#                )
#            else:
#                logger.info("No dtype row detected — reading CSV normally.")
#                d = pd.read_csv(path)
#
#            logger.info(f"read shape: {d.shape}")
#
#        except Exception as e:
#            logger.error(f"Failed reading file {path}: {e}")
#            raise
#    except:
#        logger.info((f"read file without datatypes"))
#        d = pd.read_csv(path, skiprows=[1], index_col=0)
#
#    return d

def get_substring_index(full: str, substring:str) -> Tuple[int, int]:
    start = full.find(substring)
    if start == -1:
        return -1, -1
    return start + 1, start+len(substring)


def fasta_to_dataframe(fasta_filename: Path, match: str = "") -> DataFrame:
    with fasta_filename.open() as fasta:
        logger.info(f"read fasta file {fasta_filename}")
        d = [{"ID": seq_record.id,
              "sequence": str(seq_record.seq)}
             for seq_record in SeqIO.parse(fasta, 'fasta') if str(seq_record.id).startswith(match)]

    logger.info(f"read {len(d)} fasta sequences")
    return pd.DataFrame(d)

def import_consts(exp_id: object) -> object:
    const_filename = f"GCNN_CONST_{exp_id}.py"
    const_path = ROOT_PATH_PHD_GOAL_ONE / "consts_experiment_val_set" / const_filename

    module_name = f"GCNN_CONST_{exp_id}"
    spec = importlib.util.spec_from_file_location(module_name, const_path)
    consts = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(consts)
    except Exception as e:
        print("[IMPORT ERROR]", type(e).__name__, str(e))
        raise

    cfg = vars(consts)
    for k, v in cfg.items():
        if not k.startswith("__"):
            globals()[k] = v
    return consts


def filter_Sequenceunavailable_from_fasta(fasta_file: Path):
    fasta_sequences = SeqIO.parse(fasta_file.open(), 'fasta')
    valid_seq = []
    all_cnt = 0
    valid_cnt = 0
    for s in fasta_sequences:
        all_cnt += 1
        if s.seq != 'Sequenceunavailable':
            s.id=s.id[:50] # max len for blastdb id
            valid_seq.append(s)
            valid_cnt += 1

    SeqIO.write(valid_seq, fasta_file, 'fasta')
    logger.info(f"filter_Sequenceunavailable_from_fasta \n"
                f"all seq={all_cnt} \n"
                f"valid seq={valid_cnt}")


def concatenate_biomart_df(organism: str):
    df_dict = {
        "utr3" : pd.read_csv(BIOMART_DATA_PATH/f"{organism}_3utr.csv"),
        "utr5": pd.read_csv(BIOMART_DATA_PATH / f"{organism}_5utr.csv"),
        "cds": pd.read_csv(BIOMART_DATA_PATH / f"{organism}_coding.csv")
    }
    for region in df_dict:
        df_dict[region]["region"] = region

    return pd.concat(df_dict.values(), ignore_index=True)

def call_wrapper(cmd: str, cwd: Path):
    # logger.info(f"running {cmd} from {cwd}")
    print(cmd.split())
    print(cwd.resolve())
    return subprocess.call(cmd.split(), cwd=cwd.resolve())


def DirectorySpecificBashOperator(task_id: str, cmd: str, dag: DAG, cwd: Path) -> \
        PythonOperator:
    return PythonOperator(
        task_id=task_id,
        python_callable=call_wrapper,
        op_kwargs={"cmd": cmd,
                   'cwd': cwd},
        dag=dag)


def apply_in_chunks(df: DataFrame, func: Callable, number_of_chunks: int=5):
    chunk_size = int(len(df) / number_of_chunks)
    index_chunks = chunked(df.index, chunk_size)
    return pd.concat([df.loc[ii].apply(func=func, axis=1) for ii in index_chunks], axis=0)


def split_file(infile: Path, dir: Path, number_of_chunks: int=5):
    infile = Path(infile)
    df: DataFrame = read_csv(infile)
    chunk_size = int(len(df) / number_of_chunks)
    index_chunks = chunked(df.index, chunk_size)
    for i, ii in enumerate(index_chunks):
        to_csv(df.loc[ii], dir / f"{infile.stem}{i}.csv")

def clean_directory(directory_path):
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' does not exist.")
        return

    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.remove(item_path)  # Remove files or symbolic links
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)  # Recursively remove directories
        except Exception as e:
            print(f"Failed to delete {item_path}. Reason: {e}")

    print(f"Cleaned: {directory_path}")