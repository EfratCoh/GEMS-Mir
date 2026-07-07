import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from consts.global_consts import dict_map_label_dataset, FOLDS_PATH, MERGE_DATA, Data_Embedding, number_epoch, ROOT_PATH_PHD_GOAL_ONE, NEGATIVE_DATA_PATH, POSITIVE_PATH_FEATUERS, MATRIX_DATA


def nucleotide_to_feature_16(nuc, i, seq_length, seq, max_length=30, kmer_window=2):
    """
    Returns a 16-dimensional feature vector for a nucleotide in a sequence:
    - One-hot encoding (A, C, G, U/T) → 4
    - GC content flag → 1
    - Unknown nucleotide flag (N) → 1
    - Purine / Pyrimidine flags → 2
    - Normalized position in sequence → 1
    - Is first / is last flags → 2
    - Normalized sequence length → 1
    - Wobble pairing flag (G/U) → 1
    - Frequency of the nucleotide in the sequence → 1
    - Local GC density (k-mer window) → 1
    - Local sequence entropy → 1
    """

    # One-hot encoding for known nucleotides
    mapping = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'U': [0, 0, 0, 1],
        'T': [0, 0, 0, 1],  # T treated like U
        'N': [0, 0, 0, 0]   # Unknown nucleotide
    }

    nuc = nuc.upper()
    one_hot = mapping.get(nuc, [0, 0, 0, 0])

    # Basic flags
    is_gc = 1 if nuc in ['G', 'C'] else 0
    is_unknown = 1 if nuc == 'N' else 0
    is_purine = 1 if nuc in ['A', 'G'] else 0
    is_pyrimidine = 1 if nuc in ['C', 'U', 'T'] else 0

    # Normalized position
    pos_norm = i / seq_length if seq_length > 0 else 0

    # First / Last flags
    is_first = 1 if i == 0 else 0
    is_last = 1 if i == seq_length - 1 else 0

    # Normalized sequence length
    length_norm = seq_length / max_length if max_length > 0 else 0

    # Wobble pairing potential (G/U)
    is_wobble = 1 if nuc in ['G', 'U'] else 0

    # Frequency of nucleotide in the sequence
    nuc_count = seq.upper().count(nuc) if seq_length > 0 else 0
    nuc_freq = nuc_count / seq_length if seq_length > 0 else 0

    # Local GC density (k-mer window)
    left = max(0, i - kmer_window)
    right = min(seq_length, i + kmer_window + 1)
    local_seq = seq[left:right].upper()
    gc_count = local_seq.count('G') + local_seq.count('C')
    local_gc_density = gc_count / len(local_seq) if len(local_seq) > 0 else 0

    # Local sequence entropy
    from collections import Counter
    import math

    counts = Counter(local_seq)
    probs = [count / len(local_seq) for count in counts.values()]
    entropy = -sum(p * math.log2(p) for p in probs) if len(probs) > 0 else 0

    # Final feature vector (exactly 16 features)
    return one_hot + [
        is_gc,
        is_unknown,
        is_purine,
        is_pyrimidine,
        pos_norm,
        is_first,
        is_last,
        length_norm,
        is_wobble,
        nuc_freq,
        local_gc_density,
        entropy
    ]


def add_features_to_graphs(cfg, g, sequence_data, num_mirna_nodes, num_mrna_nodes):
    """
    Adds padded biological features to each node in the graph.
    Pads each feature vector with zeros to match the required input dimension.
    """

    mirna_seq = sequence_data.get("miRNA_sequence", "")
    mrna_seq = sequence_data.get("fragment", "")
    total_nodes = num_mirna_nodes + num_mrna_nodes
    node_features = torch.zeros((total_nodes, cfg.dim_vector_in))  # preallocate

    for i in range(num_mirna_nodes):
        if i < len(mirna_seq):
            feats = nucleotide_to_feature_16(mirna_seq[i], i, len(mirna_seq), mirna_seq)

            feats_tensor = torch.tensor(feats, dtype=torch.float32)
            node_features[i, :len(feats)] = feats_tensor  # pad rest with zeros

    for i in range(num_mrna_nodes):
        idx = num_mirna_nodes + i
        if i < len(mrna_seq):
            feats = nucleotide_to_feature_16(mrna_seq[i], i, len(mrna_seq), mrna_seq)

            feats_tensor = torch.tensor(feats, dtype=torch.float32)
            node_features[idx, :len(feats)] = feats_tensor  # pad rest with zeros

    g.ndata['feature'] = node_features
    return g


def create_graph_with_features(cfg, matrices, sequence_dict, idx, weight_scale_factor=5.0):
    graphs = []
    num_graph = 0

    for matrix in matrices:
        matrix = torch.tensor(matrix, dtype=torch.float32)

        num_mirna_nodes = matrix.shape[0]
        num_mrna_nodes = matrix.shape[1]

        src = []
        dst = []
        weights = []
        edge_types = []

        # pairing (edge_type 0)
        pair_src, pair_dst = matrix.nonzero(as_tuple=True)
        for s, d in zip(pair_src.tolist(), pair_dst.tolist()):
            mi = s
            mj = d + num_mirna_nodes
            raw_w = matrix[s, d].item()
            w = raw_w * weight_scale_factor
            src += [mi, mj]
            dst += [mj, mi]
            weights += [w, w]
            edge_types += [0, 0]

        # # miRNA sequence (edge_type 1)
        for i in range(num_mirna_nodes - 1):
            src += [i, i + 1]
            dst += [i + 1, i]
            weights += [0.5 * weight_scale_factor] * 2
            edge_types += [1, 1]
        #
        # # mRNA sequence (edge_type 1)
        for i in range(num_mrna_nodes - 1):
            a = num_mirna_nodes + i
            b = num_mirna_nodes + i + 1
            src += [a, b]
            dst += [b, a]
            weights += [0.5 * weight_scale_factor] * 2
            edge_types += [1, 1]

        # self-loops (edge_type 2)
        for i in range(num_mirna_nodes + num_mrna_nodes):
            src.append(i)
            dst.append(i)
            weights.append(0.1 * weight_scale_factor)
            edge_types.append(2)

        # graph construction
        g = dgl.graph((src, dst), num_nodes=num_mirna_nodes + num_mrna_nodes)
        g.edata['weight'] = torch.tensor(weights, dtype=torch.float32)
        g.edata['edge_type'] = torch.tensor(edge_types, dtype=torch.int64)

        # node features
        g = add_features_to_graphs(cfg, g, sequence_dict[idx[num_graph]], num_mirna_nodes, num_mrna_nodes)

        graphs.append(g)
        num_graph += 1

    return graphs
