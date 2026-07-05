from typing import Tuple
import RNA
from BreathesDuplex.Duplex import Duplex

# from Duplex import Duplex
import subprocess
import os
# from utils.utilsfile import read_csv, to_csv


def find_pairing(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]

class ViennaDuplexBreath(Duplex):
    @classmethod
    def createDuplex(cls, mirna: str, target: str):
        mirna = mirna.replace('T', 'U').upper()
        target = target.replace('T', 'U').upper()

        duplex_dict_result = {}
        duplex_dict = RNA.duplex_subopt(mirna, target, w=0, delta=1500)
        count_duplex = 0
        for duplex in duplex_dict:
            count_duplex = count_duplex +1
            (mir_pairing, mrna_pairing) = duplex.structure.split('&')
            MFE = duplex.energy
            mir_coor = (duplex.i - len(mir_pairing), duplex.i)
            # The target sequences
            mrna_coor = (duplex.j - 1, duplex.j + len(mrna_pairing) - 1)
            # print("func" , mrna_coor)
            active_mir = mirna[mir_coor[0]:mir_coor[1]]
            active_mrna = target[mrna_coor[0]:mrna_coor[1]]
            # print(active_mrna)
            mir_idx = find_pairing(mir_pairing, '(')
            mrna_idx = find_pairing(mrna_pairing, ')')
            mrna_idx = mrna_idx[::-1]
            mrna_len = len(active_mrna)

            mrna_bulge = ""
            mrna_inter = ""
            mir_inter = ""
            mir_bulge = ""

            mir_i = 0
            mrna_i = mrna_len - 1
            if mir_coor[0] > 0:
                mir_bulge += mirna[:mir_coor[0]]
                mir_inter += " " * mir_coor[0]
                mrna_inter += " " * mir_coor[0]
                mrna_addition_len = mrna_coor[1] + mir_coor[0] - mrna_coor[1]
                mrna_bulge_additon = target[mrna_coor[1]:mrna_coor[1] + mir_coor[0]]
                mrna_bulge_additon = mrna_bulge_additon + "#" * (mrna_addition_len - len(mrna_bulge_additon))
                mrna_bulge += mrna_bulge_additon[::-1]

            for i in range(len(mir_idx)):
                # deal with the bulge
                mir_bulge_idx = range(mir_i, mir_idx[i])
                mir_bulge += active_mir[mir_i:mir_idx[i]]
                mrna_bulge_idx = range(mrna_i, mrna_idx[i], -1)
                mrna_bulge += active_mrna[mrna_i:mrna_idx[i]: -1]
                c_pos = max(len(mrna_bulge_idx), len(mir_bulge_idx))
                mrna_inter += " " * c_pos
                mir_inter += " " * c_pos
                mrna_bulge += " " * (c_pos - len(mrna_bulge_idx))
                mir_bulge += " " * (c_pos - len(mir_bulge_idx))

                # deal with the interaction
                mir_bulge += " "
                mir_inter += active_mir[mir_idx[i]]
                mrna_bulge += " "
                mrna_inter += active_mrna[mrna_idx[i]]
                # update the idx
                mir_i = mir_idx[i] + 1
                mrna_i = mrna_idx[i] - 1

            mir_i += mir_coor[0]
            mir_bulge_additon = mirna[mir_i:]
            mir_bulge += mir_bulge_additon

            mrna_addition = target[max(0, mrna_coor[0] - len(mir_bulge_additon) + mrna_i + 1):mrna_coor[0] + mrna_i + 1]
            mrna_bulge += mrna_addition[::-1]

            new_interaction = {count_duplex : (mrna_bulge, mrna_inter, mir_inter, mir_bulge, MFE)}

            duplex_dict_result.update(new_interaction)
            # print(new_interaction)
            # print(duplex.energy)
            print(mir_bulge)
            print(mir_inter)
            print(mrna_inter)
            print(mrna_bulge)
            # dp = Duplex.fromStrings(mrna_bulge, mrna_inter, mir_inter, mir_bulge)
            # print("canon full:", dp.canonical_seed)
            # print("non canon full:", dp.noncanonical_seed)
            # print("########################################################")
        # print(count_duplex)
        return duplex_dict_result


# mirna = "UAUUGCACUUGUCCCGGCCUGU"
# target = "AUGGGCCCGGAGACUACUGCAAAG"
# fragment = "UUUUCCAGAUUGAUGGGCCCGGAGACUACUGCAAAGACUAUAGUUUUGGUUAAAAAUGUUCUUUCCCGACAUUAUGUUCAUC"
#
# #
# mirna = "UUCAAGUAAUCCAGGAUAGGCU"
# fragment = "UUUAUCCUGGAUUAACUUAGAUAACUUUUGUAGCAGUGGUUAUAUUGCUUAUAAUUUAAUGUACAAUACUAUUGA"
# # site = "GUAUUUCCAGGGAUUAGGAUUA"
# mirna = mirna.replace('T', 'U').upper()
# # site = site.replace('T', 'U').upper()
# target = fragment.replace('T', 'U').upper()
# duplex_dict_result = {}
#
# duplex_dict = RNA.duplex_subopt(mirna, fragment, w=0, delta=1500)
#
# count_duplex = 0
# for duplex in duplex_dict:
#     count_duplex = count_duplex + 1
#     (mir_pairing, mrna_pairing) = duplex.structure.split('&')
#     mir_coor = (duplex.i - len(mir_pairing), duplex.i)
#     MFE = duplex.energy
#
# #     # The target sequences
#     mrna_coor = (duplex.j - 1, duplex.j + len(mrna_pairing) - 1)
#     # print("func" , mrna_coor)
#     active_mir = mirna[mir_coor[0]:mir_coor[1]]
#     active_mrna = target[mrna_coor[0]:mrna_coor[1]]
#     # print(active_mrna)
#     mir_idx = find_pairing(mir_pairing, '(')
#     mrna_idx = find_pairing(mrna_pairing, ')')
#     mrna_idx = mrna_idx[::-1]
#     mrna_len = len(active_mrna)
#
#     mrna_bulge = ""
#     mrna_inter = ""
#     mir_inter = ""
#     mir_bulge = ""
#
#     mir_i = 0
#     mrna_i = mrna_len - 1
#     if mir_coor[0] > 0:
#         mir_bulge += mirna[:mir_coor[0]]
#         mir_inter += " " * mir_coor[0]
#         mrna_inter += " " * mir_coor[0]
#         mrna_addition_len = mrna_coor[1] + mir_coor[0] - mrna_coor[1]
#         mrna_bulge_additon = target[mrna_coor[1]:mrna_coor[1] + mir_coor[0]]
#         mrna_bulge_additon = mrna_bulge_additon + "#" * (mrna_addition_len - len(mrna_bulge_additon))
#         mrna_bulge += mrna_bulge_additon[::-1]
#
#     for i in range(len(mir_idx)):
#         # deal with the bulge
#         mir_bulge_idx = range(mir_i, mir_idx[i])
#         mir_bulge += active_mir[mir_i:mir_idx[i]]
#         mrna_bulge_idx = range(mrna_i, mrna_idx[i], -1)
#         mrna_bulge += active_mrna[mrna_i:mrna_idx[i]: -1]
#         c_pos = max(len(mrna_bulge_idx), len(mir_bulge_idx))
#         mrna_inter += " " * c_pos
#         mir_inter += " " * c_pos
#         mrna_bulge += " " * (c_pos - len(mrna_bulge_idx))
#         mir_bulge += " " * (c_pos - len(mir_bulge_idx))
#
#         # deal with the interaction
#         mir_bulge += " "
#         mir_inter += active_mir[mir_idx[i]]
#         mrna_bulge += " "
#         mrna_inter += active_mrna[mrna_idx[i]]
#         # update the idx
#         mir_i = mir_idx[i] + 1
#         mrna_i = mrna_idx[i] - 1
#
#     mir_i += mir_coor[0]
#     mir_bulge_additon = mirna[mir_i:]
#     mir_bulge += mir_bulge_additon
#
#     mrna_addition = target[max(0, mrna_coor[0] - len(mir_bulge_additon) + mrna_i + 1):mrna_coor[0] + mrna_i + 1]
#     mrna_bulge += mrna_addition[::-1]
#
#     new_interaction = {count_duplex: (mrna_bulge, mrna_inter, mir_inter, mir_bulge)}
#
#     duplex_dict_result.update(new_interaction)
#     # print(new_interaction)
#     print(mir_bulge, "mirna 5-3")
#     print(mir_inter)
#     print(mrna_inter)
#     print(mrna_bulge, "mrna 3-5")
#     if int(MFE) == int(-20.7):
#         print("g")
#
#         dp = Duplex.fromStrings(mrna_bulge, mrna_inter, mir_inter, mir_bulge)
#         print("canon full:", dp.canonical_seed)
#         print("non canon full:", dp.noncanonical_seed)
#

#
