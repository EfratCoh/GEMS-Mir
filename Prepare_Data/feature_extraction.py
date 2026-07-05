from pathlib import Path
from features.AccessibilityFeatures import AccessibilityFeatures
from features.EnergyFeatures import EnergyFeatures
from features.MatchingFeatures import MatchingFeatures
from features.MrnaFeatures import MrnaFeatures
from features.SeedFeatures import SeedFeatures
from utils.logger import logger
from utils.utilsfile import apply_in_chunks, get_wrapper, read_csv, to_csv
from pandas import Series, DataFrame
from BreathesDuplex.Duplex import Duplex
import pandas as pd
from consts.global_consts import ROOT_PATH

FEATURE_CLASSES = [SeedFeatures, MatchingFeatures, MrnaFeatures, EnergyFeatures, AccessibilityFeatures]


def row_feature_extractor(miRNA: str, site: str, start: int, end: int, full_mrna: str,
                          mrna_bulge: str, mrna_inter: str, mir_inter: str, mir_bulge: str) -> Series:

    dp = Duplex.fromStrings(mrna_bulge, mrna_inter, mir_inter, mir_bulge)
    f = [feature_cls(dp, miRNA, site, start, end, full_mrna).get_features() for feature_cls in FEATURE_CLASSES]
    return pd.concat(f)


def df_feature_extractor(valid_df: DataFrame) -> DataFrame:
    return valid_df.apply(func=get_wrapper(row_feature_extractor,
                                           "miRNA_sequence", "site", "start", "end", "full_mrna",
                                           'mrna_bulge', 'mrna_inter', 'mir_inter', 'mir_bulge'), axis=1)


def feature_extraction(in_df):
    valid_df = in_df.query("valid_row & duplex_valid")
    feature_df = df_feature_extractor(valid_df)
    result = pd.merge(left=valid_df, right=feature_df, left_index=True, right_index=True, how='left')
    # not save only canon and non canon
    # print("Number of interactions before canon and no cannon:", result.shape[0])
    # result = result[(result["Seed_match_canonical"] == True) | (result["Seed_match_noncanonical"] == True)]
    # print("Number of interactions before after canon and no cannon:", result.shape[0])
    result.rename(columns={'full_mrna': 'sequence'}, inplace=True)
    return result




