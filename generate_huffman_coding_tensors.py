import argparse

# miscellaneous
import builtins
import datetime
import json
import sys
import time
import random

# onnx
# The onnx import causes deprecation warnings every time workers
# are spawned during testing. So, we filter out those warnings.
import warnings

# data generation
import huffman_coding as hc

import torch

# numpy
import numpy as np
import sklearn.metrics
import pickle
import os

from typing import Optional
from collections import defaultdict


def run_huffman_coding(
    feature_frequency_mappings_path: str,
    huffman_coding_tensors_path: str,
    n: int,
    use_bell_curve: bool,
    test_conflict_ratio: bool,
    use_reverse_and_shift: bool
) -> None:
    if not os.path.exists(feature_frequency_mappings_path):
        print("feature_frequency_mappings_path is invalid")
        exit(1)
    
    sys.setrecursionlimit(5000)
    torch.set_printoptions(profile="full")
    
    if use_bell_curve:
        print("use_bell_curve")
    else:
        print("not use_bell_curve")
    
    if test_conflict_ratio:
        print("test_conflict_ratio")
    else:
        print("not test_conflict_ratio")

    if use_reverse_and_shift:
        print("use_reverse_and_shift")
    else:
        print("not use_reverse_and_shift")
        
    # with open("huffman_coding_feature_frequency_mappings_12132023.pkl", 'rb') as file:
    with open(feature_frequency_mappings_path, 'rb') as file:
        feature_frequency_mappings = pickle.load(file)

    num_embs = len(feature_frequency_mappings)
    huffman_coding_dicts = [{} for _ in range(num_embs)]
    max_lens = [0 for _ in range(num_embs)]
    huffman_coding_tensors = []

    for i in range(num_embs):
        if use_bell_curve:
            for k in feature_frequency_mappings[i].keys():
                feature_frequency_mappings[i][k] = -feature_frequency_mappings[i][k]
            update_freq_dict = feature_frequency_mappings[i]
            if i == 1:
                print(f"feature_frequency_mappings{i}: {feature_frequency_mappings[i]}")
                print(f"update_freq_dict: {update_freq_dict}")
        else:
            update_freq_dict = feature_frequency_mappings[i]

        huffman_coding_dicts[i], max_lens[i] = hc.n_ary_huffman_coding(update_freq_dict, n)
        print("max_lens[", i, "]:", max_lens[i])
        n_ary_huffman_coding_tensor = hc.generate_n_ary_huffman_coding_tensor(huffman_coding_dicts[i], max_lens[i], len(huffman_coding_dicts[i]), max_lens[i], n)
        
        if use_reverse_and_shift:
            n_ary_huffman_coding_tensor = torch.flip(n_ary_huffman_coding_tensor, dims=[1])
            nonzero_mask = n_ary_huffman_coding_tensor != 0
            nonzero_indices = torch.argmax(nonzero_mask.to(torch.int), dim=1)
            row_indices = torch.arange(n_ary_huffman_coding_tensor.size(0)).reshape(-1, 1)
            first_nonzero_indices = torch.cat((row_indices, nonzero_indices.unsqueeze(1)), dim=1)
            selected_elements = n_ary_huffman_coding_tensor[first_nonzero_indices[:, 0], first_nonzero_indices[:, 1]]
            broadcasted_tensor = selected_elements.unsqueeze(0).expand(n_ary_huffman_coding_tensor.size(1),-1)
            broadcasted_tensor=broadcasted_tensor.T
            n_ary_huffman_coding_tensor[nonzero_mask]-=broadcasted_tensor[nonzero_mask]
            mask = torch.zeros_like(n_ary_huffman_coding_tensor, dtype=torch.bool)
            mask[first_nonzero_indices[:, 0], first_nonzero_indices[:, 1]] = True
            n_ary_huffman_coding_tensor[mask]+=broadcasted_tensor[mask]
            n_ary_huffman_coding_tensor %= (n * max_lens[i] + 1)

        if test_conflict_ratio:
            embedding_index_counter = defaultdict(int)
            print(f"n_ary_huffman_coding_tensor[0]: {n_ary_huffman_coding_tensor[0]}")
            for k, v in huffman_coding_dicts[i].items():
                for j in n_ary_huffman_coding_tensor[k]:
                    if j != 0:
                        embedding_index_counter[j.item()] += update_freq_dict[k]
            total_frequency = 0
            for k, v in sorted(embedding_index_counter.items(), key=lambda x: x[1]):
                print(f"index: {k} frequency: {v}")
                total_frequency += v
            print(f"total_frequency: {total_frequency}")

        huffman_coding_tensors.append(n_ary_huffman_coding_tensor)
    
    if not test_conflict_ratio:
        with open(huffman_coding_tensors_path, 'wb') as file:
            pickle.dump(huffman_coding_tensors, file)
            pickle.dump(max_lens, file)

        with open(huffman_coding_tensors_path, 'rb') as file:
            huffman_coding_tensors = pickle.load(file)
            max_lens = pickle.load(file)


def run_index_access_frequency_distribution(
    feature_frequency_mappings_path: str,
    huffman_coding_tensors_path: str,
    n: int,
    use_bell_curve: bool
) -> None:
    if not os.path.exists(feature_frequency_mappings_path):
        print("feature_frequency_mappings_path is invalid")
        exit(1)
    
    sys.setrecursionlimit(5000)
    torch.set_printoptions(profile="full")
    
    if use_bell_curve:
        print("use_bell_curve")
    else:
        print("not use_bell_curve")

    with open(feature_frequency_mappings_path, 'rb') as file:
        feature_frequency_mappings = pickle.load(file)

    num_embs = len(feature_frequency_mappings)
    huffman_coding_dicts = [{} for _ in range(num_embs)]
    max_lens = [0 for _ in range(num_embs)]
    huffman_coding_tensors = []

    for i in range(num_embs):
        if i != 6:
            continue

        if use_bell_curve:
            for k in feature_frequency_mappings[i].keys():
                feature_frequency_mappings[i][k] = -feature_frequency_mappings[i][k]
            update_freq_dict = feature_frequency_mappings[i]
            if i == 1:
                print(f"feature_frequency_mappings{i}: {feature_frequency_mappings[i]}")
                print(f"update_freq_dict: {update_freq_dict}")
        else:
            update_freq_dict = feature_frequency_mappings[i]

        huffman_coding_dicts[i], max_lens[i] = hc.n_ary_huffman_coding(update_freq_dict, n)
        print("max_lens[", i, "]:", max_lens[i])
        n_ary_huffman_coding_tensor = hc.generate_n_ary_huffman_coding_tensor(huffman_coding_dicts[i], max_lens[i], len(huffman_coding_dicts[i]), max_lens[i], n)
        embedding_index_counter = defaultdict(int)
        for k, v in huffman_coding_dicts[i].items():
            for j in n_ary_huffman_coding_tensor[k]:
                if j != 0:
                    embedding_index_counter[j.item()] += update_freq_dict[k]
        for k, v in sorted(embedding_index_counter.items()):
            print(f"index: {k} frequency: {v}")
        huffman_coding_tensors.append(n_ary_huffman_coding_tensor)


def run_huffman_coding_debug(feature_frequency_mappings_path: str, huffman_coding_tensors_path: str, n: Optional[int] = 2) -> None:
    sys.setrecursionlimit(5000)
    torch.set_printoptions(profile="full")
    if not os.path.exists(feature_frequency_mappings_path):
        print("feature_frequency_mappings_path is invalid")
        exit(1)

    with open(feature_frequency_mappings_path, 'rb') as file:
        feature_frequency_mappings = pickle.load(file)

    num_embs = len(feature_frequency_mappings)
    huffman_coding_dicts = [{} for _ in range(num_embs)]
    max_lens = [0 for _ in range(num_embs)]
    huffman_coding_tensors = []
    embedding_table_index = 0

    N = len(feature_frequency_mappings[embedding_table_index])
    print("N:", N)
    update_freq_dict = feature_frequency_mappings[embedding_table_index]

    huffman_coding_dicts[embedding_table_index], max_lens[embedding_table_index] = hc.n_ary_huffman_coding(update_freq_dict, n)
    print("max_lens[", embedding_table_index, "]:", max_lens[embedding_table_index])
    huffman_coding_tensors.append(hc.generate_n_ary_huffman_coding_tensor(huffman_coding_dicts[embedding_table_index], max_lens[embedding_table_index], len(huffman_coding_dicts[embedding_table_index]), max_lens[embedding_table_index], n))
    for embedding_index in range(N):
        print(f"{embedding_index} frequency: {update_freq_dict[embedding_index]}, \n {huffman_coding_tensors[0][embedding_index]}")
    
    with open(huffman_coding_tensors_path, 'wb') as file:
        pickle.dump(huffman_coding_tensors, file)
        pickle.dump(max_lens, file)

    with open(huffman_coding_tensors_path, 'rb') as file:
        huffman_coding_tensors = pickle.load(file)
        max_lens = pickle.load(file)
        print("huffman_coding_tensors.size()=", len(huffman_coding_tensors))


def run():
    ### parse arguments ###
    parser = argparse.ArgumentParser(
        description="Train Deep Learning Recommendation Model (DLRM)"
    )
    parser.add_argument("--feature-frequency-mappings-path", type=str, default="")
    parser.add_argument("--huffman-coding-tensors-path", type=str, default="")
    parser.add_argument("--huffman-coding-digits", type=int, default=2)
    parser.add_argument("--use-bell-curve", action="store_true", default=False)
    parser.add_argument("--test-conflict-ratio", action="store_true", default=False)
    parser.add_argument("--use-reverse-and-shift", action="store_true", default=False)

    global args
    args = parser.parse_args()

    print(f"args.feature_frequency_mappings_path: {args.feature_frequency_mappings_path}")
    print(f"args.huffman_coding_tensors_path: {args.huffman_coding_tensors_path}")
    print(f"args.huffman_coding_digits: {args.huffman_coding_digits}")
    print(f"args.use_bell_curve: {args.use_bell_curve}")
    print(f"args.test_conflict_ratio: {args.test_conflict_ratio}")
    print(f"args.use_reverse_and_shift: {args.use_reverse_and_shift}")
    
    program_start_time = start_time = time.time()

    run_huffman_coding(
        args.feature_frequency_mappings_path,
        args.huffman_coding_tensors_path,
        args.huffman_coding_digits,
        args.use_bell_curve,
        args.test_conflict_ratio,
        args.use_reverse_and_shift
    )

    print(f"program takes {time.time() - program_start_time} seconds", flush=True)

if __name__ == "__main__":
    run()
