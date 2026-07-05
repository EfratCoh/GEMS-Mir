from pathlib import Path
from BreathesDuplex.RNADuplex_Variations import ViennaDuplexBreath

# Path for master project
ROOT_PATH = Path("/home/efrco/efrco-master/")
log_file = ROOT_PATH / "pipeline_log.log"
BIOMART_PATH = Path("/sise/vaksler-group/IsanaRNA/miRNA_target_rules/benorgi/pipeline/data/biomart/")
DATA_PATH = Path("/sise/vaksler-group/IsanaRNA/miRNA_target_rules/benorgi/pipeline/")
POSITIVE_PATH_DATA = Path("/sise/vaksler-group/IsanaRNA/miRNA_target_rules/benorgi/pipeline/data/pipeline_steps/features/")
MERGE_DATA = ROOT_PATH / "data/positive_interactions/"
NEGATIVE_DATA_PATH = Path("/home/efrco/efrco-master/data/negative_interactions/")
DATA_PATH_INTERACTIONS = Path("/home/efrco/efrco-master/data/")
GENERATE_DATA_PATH= ROOT_PATH / "generate_interactions/"
CLIP_PATH_DATA = Path("/sise/vaksler-group/IsanaRNA/miRNA_target_rules/Isana/")
POSITIVE_PATH_FEATUERS = ROOT_PATH / "data/positive_interactions/positive_interactions_new/featuers_step/"
Data_Embedding = Path("/home/efrco/PHD/Goal_One/Data_embedding/")
# Path for PHD project
ROOT_PATH_PHD_GOAL_ONE = Path("/home/efrco/PHD/Goal_One/")
FOLDS_PATH = Path("/home/efrco/PHD/Goal_One/Data_Folds/")
MATRIX_DATA= Path("/home/efrco/PHD/Goal_One/Data_Graph_Matrix/")
DATA_TRAIN_TEST = Path( "/home/efrco/PHD/Goal_One/Data_merge_embedding_features/")

SITE_EXTRA_CHARS: int = 3
HUMAN_SITE_EXTENDED_LEN: int = 25
MINIMAL_BLAST_COVERAGE = 95
MINIMAL_BLAST_IDENTITY = 95
MINIMAL_LENGTH_TO_BLAST = 10

DUPLEX_DICT = {"ViennaDuplex": ViennaDuplexBreath}


CONFIG = {
    'minimum_pairs_for_interaction': 11,
    'duplex_method' :  ["vienna"],
    'max_process' : 1
}

dict_map_label_dataset = {"darnell_human_dataset":0, "NPS_darnell_human_dataset":1}

#####################################
number_epoch = 40
number_fold = 10
