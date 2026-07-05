from pathlib import Path
from pandas import DataFrame, Series
import pandas as pd
from consts.global_consts import HUMAN_SITE_EXTENDED_LEN, ROOT_PATH, BIOMART_PATH, GENERATE_DATA_PATH
from consts.global_consts import DUPLEX_DICT
from BreathesDuplex.Duplex import Duplex
from utils.logger import logger
from utils.utilsfile import get_wrapper, read_csv, to_csv
from utils.logger import logger
from utils.utilsfile import get_wrapper, read_csv, to_csv


def do_duplex(miRNA_ID, miRNA_sequence, site, full_mrna, Gene_ID, ID_interaction, cls: Duplex, path_dir_target, method) -> Series:
    columns = ["miRNA ID", "miRNA_sequence",  "full_mrna", "Gene_ID", "ID_interaction","fragment"]
    values = [miRNA_ID, miRNA_sequence, full_mrna, Gene_ID,ID_interaction, site]
    row = {column: value for column, value in zip(columns, values)}
    df = pd.DataFrame([row])

    mirna = miRNA_sequence
    target = site

    if pd.isna(mirna) or pd.isna(target):
        return Series({"duplex_valid" : False,
                       "not_match_site": "",
                           "site": "",
                       "fragment": "",
                        "mrna_bulge": "",
                      "mrna_inter": "",
                      "mir_inter": "",
                      "mir_bulge": "",
                       'ID_interaction':ID_interaction,
                       "Seed_match_canonical": "",
                       "Seed_match_noncanonical": ""
                       })
    dp_dict = cls.fromChimera(mirna, target)


    series_list = []

    for key, dp in dp_dict.items():

        series_obj = Series({"duplex_valid": dp.valid,
                       "not_match_site": dp.site_non_match_tail,
                       "site": dp.site[::-1],
                    "fragment": site,
                  "mrna_bulge": dp.mrna_bulge,
                  "mrna_inter": dp.mrna_inter,
                  "mir_inter": dp.mir_inter,
                  "mir_bulge": dp.mir_bulge,
                'ID_interaction': ID_interaction,
                "Seed_match_canonical": dp.canonical_seed,
                "Seed_match_noncanonical":dp.noncanonical_seed })
        series_list.append(series_obj)
    print(series_list)
    series_duplex = pd.DataFrame(series_list)

    result = pd.merge(left=df, right=series_duplex,on='ID_interaction')
    result = result[result['duplex_valid'] == True]

    result["duplex_method"] = method
    ID_interaction = ID_interaction + ".csv"
    fout = path_dir_target / ID_interaction
    to_csv(result, Path(fout))
    return series_list



def duplex(method: str, fin: str, fout: str):

    duplex_cls: Duplex = DUPLEX_DICT[method]
    logger.info(f"{method} do_duplex to {fin}")
    in_df: DataFrame = read_csv(Path(fin))
    seq_cols = ['miRNA sequence', 'full_mrna', 'site']
    in_df[seq_cols] = in_df[seq_cols].replace(to_replace='T', value='U', regex=True)

    in_df.apply(func=get_wrapper(do_duplex, "miRNA ID", "miRNA sequence",'site', 'full_mrna', 'Gene_ID', 'ID_interaction',
         cls=duplex_cls, path_dir_target=fout, method = method), axis=1)



# if __name__ == '__main__':
#     # cli()
#     fin = ROOT_PATH / "generate_interactions/new_1.csv"
#     fout =ROOT_PATH / "generate_interactions/duplex.csv"
#     duplex('ViennaDuplex', fin, fout)
