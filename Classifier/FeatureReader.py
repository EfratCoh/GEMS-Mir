from pathlib import Path
from typing import List
from numpy import ndarray
from pandas import DataFrame
# from utilsfile import read_csv, to_csv
from utils.utilsfile import read_csv, to_csv, import_consts
import pandas as pd

class FeatureReader:
    def __init__(self, expected_num_of_features: int):
        self.exp_id = 0
        self.expected_num_of_features = expected_num_of_features

    def _feature(self, data) -> List:
        col_list = list(data.columns)
        columns_to_remove = ["organism", "Seed_match_canonical", "Seed_match_noncanonical", "Label", "GCN_Prob_Positive", "True_Label"]
        col_list = [col for col in col_list if col not in columns_to_remove]
        all_features = col_list[col_list.index("Seed_match_interactions_all"):]
        return all_features

    def file_reader(self, in_file: Path) -> (DataFrame, ndarray):
        data: DataFrame = read_csv(in_file)

        # we care on object type columns
        for i in range(1, 21):
            name_col = "miRNAMatchPosition_" + str(i)
            data[name_col] = data[name_col].astype('category').cat.codes
        data["microRNA_name"] = data[name_col].astype('category').cat.codes
        data.drop(columns=['microRNA_name'], inplace=True)

        return self.df_reader(data)

    def df_reader(self, in_df: DataFrame) -> (DataFrame, ndarray):
        in_df.rename(columns={"label": "Label"}, inplace=True)

        y: ndarray = in_df.Label.ravel()
        feature_list: List = self._feature(in_df)
        X = in_df[feature_list]
        return X, y


class AllWithoutEmbeddingReader(FeatureReader):
    def __init__(self):
        super().__init__(500)

    def setnumbrtexperiments(self, exp_id):
        self.exp_id = 0

    def _feature(self, data) -> List:
        cfg = import_consts(self.exp_id)
        all_features = super()._feature(data)
        all_features = [f for f in all_features if not str(f).startswith("HotPairing")]
        all_features =  [f for f in all_features if not str(f).startswith("dim")]

        assert len(all_features) == 500 + cfg.dim_vector_out, f"All feature read error. Read: {len(all_features)}"
        return all_features


class AllWithEmbedding(FeatureReader):
    def __init__(self):
        super().__init__(500)
    def setnumbrtexperiments(self, exp_id):
        self.exp_id = exp_id
        cfg = import_consts(exp_id)
        self.expected_num_of_features = 500 + cfg.dim_vector_out
    def _feature(self, data) -> List:

        cfg = import_consts(self.exp_id)
        all_features = super()._feature(data)
        all_features = [f for f in all_features if not str(f).startswith("HotPairing")]
        return all_features

class OnlyWithEmbedding(FeatureReader):
    def __init__(self):
        super().__init__(0)
    def setnumbrtexperiments(self, exp_id):
        self.exp_id = exp_id
        cfg = import_consts(exp_id)
        self.expected_num_of_features = cfg.dim_vector_out
    def _feature(self, data) -> List:
        cfg = import_consts(self.exp_id)
        all_features = super()._feature(data)
        all_features = [f for f in all_features if str(f).startswith("dim")]
        return all_features

class AllFeatursReader(FeatureReader):
    def __init__(self):
        super().__init__(580)

    def _feature(self, data) -> List:
        all_features = super()._feature(data)
        return all_features


class HotEncodingReader(FeatureReader):
    def __init__(self):
        super().__init__(90)

    def _feature(self, data) -> List:
        all_features = super()._feature(data)
        return [f for f in all_features if str(f).startswith("HotPairing")]
reader_dict = {
    "all": AllFeatursReader(),
    "hot_encoding": HotEncodingReader(),
    "without_embedding_vector": AllWithoutEmbeddingReader(),
    "with_embedding_vector": AllWithEmbedding(),
    "only_embedding_vector": OnlyWithEmbedding()

}

reader_selection_parameter = None

def get_reader() -> FeatureReader:
    assert reader_selection_parameter is not None, "reader selection parameter is none"
    return reader_dict[reader_selection_parameter]
