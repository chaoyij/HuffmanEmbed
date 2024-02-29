from __future__ import absolute_import, division, print_function, unicode_literals

import argparse

# miscellaneous
import builtins
import datetime
import json
import sys
import time
import random
import numpy as np
from collections import defaultdict
from typing import List, Dict

# onnx
# The onnx import causes deprecation warnings every time workers
# are spawned during testing. So, we filter out those warnings.
import warnings

# data generation
import dlrm_data_pytorch as dp
import huffman_coding as hc

# For distributed run
# import extend_distributed as ext_dist
# import mlperf_logger

# numpy
import numpy as np
import sklearn.metrics
import pickle

# pytorch
import torch
import torch.nn as nn
from torch._ops import ops
from torch.autograd.profiler import record_function
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import gather, scatter
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import _LRScheduler
# import optim.rwsadagrad as RowWiseSparseAdagrad
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# mixed-dimension trick
from tricks.md_embedding_bag import PrEmbeddingBag, md_solver

# quotient-remainder trick
from tricks.qr_embedding_bag import QREmbeddingBag

def dash_separated_ints(value):
    vals = value.split("-")
    for val in vals:
        try:
            int(val)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "%s is not a valid dash separated list of ints" % value
            )

    return value


def dash_separated_floats(value):
    vals = value.split("-")
    for val in vals:
        try:
            float(val)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "%s is not a valid dash separated list of floats" % value
            )

    return value


def unpack_batch(b):
    # Experiment with unweighted samples
    return b[0], b[1], b[2], b[3], torch.ones(b[3].size()), None


def print_feature_frequency_mappings(feature_frequency_mappings: List[Dict[int, int]]) -> None:
    for i, feature_frequency_mappings in enumerate(feature_frequency_mappings):
        sorted_dict_desc = dict(sorted(feature_frequency_mappings.items(), key=lambda item: item[1], reverse=True))
        print(f"Sorted dictionary {i} by values (descending): {sorted_dict_desc}")


def run():
    ### parse arguments ###
    parser = argparse.ArgumentParser(
        description="Train Deep Learning Recommendation Model (DLRM)"
    )
    parser.add_argument("--hash-flag", action="store_true", default=False)
    parser.add_argument("--bucket-flag", action="store_true", default=False)
    parser.add_argument("--sketch-flag", action="store_true", default=False)
    parser.add_argument("--compress-rate", type=float, default=0.001)
    parser.add_argument("--hc-threshold", type=int, default=200)
    parser.add_argument("--hash-rate", type=float, default=0.5)
    # data
    parser.add_argument("--data-size", type=int, default=1)
    parser.add_argument("--num-batches", type=int, default=0)
    parser.add_argument(
        "--data-generation", type=str, default="random"
    )  # synthetic or dataset
    parser.add_argument(
        "--rand-data-dist", type=str, default="uniform"
    )  # uniform or gaussian
    parser.add_argument("--rand-data-min", type=float, default=0)
    parser.add_argument("--rand-data-max", type=float, default=1)
    parser.add_argument("--rand-data-mu", type=float, default=-1)
    parser.add_argument("--rand-data-sigma", type=float, default=1)
    parser.add_argument("--data-trace-file", type=str, default="./input/dist_emb_j.log")
    parser.add_argument("--data-set", type=str, default="kaggle")  # or terabyte
    parser.add_argument("--raw-data-file", type=str, default="")
    parser.add_argument("--processed-data-file", type=str, default="")
    parser.add_argument("--data-randomize", type=str, default="total")  # or day or none
    parser.add_argument("--data-trace-enable-padding", type=bool, default=False)
    parser.add_argument("--max-ind-range", type=int, default=-1)
    parser.add_argument("--data-sub-sample-rate", type=float, default=0.0)  # in [0, 1]
    parser.add_argument("--num-indices-per-lookup", type=int, default=10)
    parser.add_argument("--num-indices-per-lookup-fixed", type=bool, default=False)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--memory-map", action="store_true", default=False)
    # training
    parser.add_argument("--mini-batch-size", type=int, default=1)
    parser.add_argument("--nepochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--print-precision", type=int, default=5)
    parser.add_argument("--numpy-rand-seed", type=int, default=123)
    parser.add_argument("--sync-dense-params", type=bool, default=True)
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument(
        "--dataset-multiprocessing",
        action="store_true",
        default=False,
        help="The Kaggle dataset can be multiprocessed in an environment \
                        with more than 7 CPU cores and more than 20 GB of memory. \n \
                        The Terabyte dataset can be multiprocessed in an environment \
                        with more than 24 CPU cores and at least 1 TB of memory.",
    )
    parser.add_argument("--test-mini-batch-size", type=int, default=-1)
    parser.add_argument("--test-num-workers", type=int, default=-1)
    parser.add_argument("--mlperf-logging", action="store_true", default=False)
    parser.add_argument("--mlperf-bin-loader", action="store_true", default=False)
    parser.add_argument("--mlperf-bin-shuffle", action="store_true", default=False)
    parser.add_argument("--feature-frequency-mappings-path", type=str, default="")
    parser.add_argument("--cat-path", type=str,
                        default="../criteo_24days/sparse")
    parser.add_argument("--dense-path", type=str,
                        default="../criteo_24days/dense")
    parser.add_argument("--label-path", type=str,
                        default="../criteo_24days/label")
    parser.add_argument("--count-path", type=str,
                        default="../criteo_24days/processed_count.bin")

    global args
    args = parser.parse_args()

    if len(args.feature_frequency_mappings_path) == 0:
        print("feature_frequency_mappings_path cannot be empty")
        exit(1)

    train_data, train_ld, test_data, test_ld = dp.make_criteo_data_and_loaders(args)
    ln_emb = train_data.counts

    feature_frequency_mappings = [defaultdict(int) for _ in range(ln_emb.size)]
    max_lens = [0 for _ in range(ln_emb.size)]

    program_start_time = start_time = time.time()

    for j, inputBatch in enumerate(train_ld):
        X, lS_o, lS_i, T, W, CBPP = unpack_batch(inputBatch)
        for k, sparse_index_group_batch in enumerate(lS_i):
            for sparse_index in sparse_index_group_batch:
                feature_frequency_mappings[k][sparse_index.item()] += 1
        if (j + 1) % 10000 == 0:
            huffman_end_time = time.time()
            huffman_time_duration = (huffman_end_time - start_time)
            print(f"huffman {j - 9999}-{j} train batches takes {huffman_time_duration} seconds", flush=True)
            start_time = time.time()

    start_time = time.time()
    for j, inputBatch in enumerate(test_ld):
        X, lS_o, lS_i, T, W, CBPP = unpack_batch(inputBatch)
        for k, sparse_index_group_batch in enumerate(lS_i):
            for sparse_index in sparse_index_group_batch:
                feature_frequency_mappings[k][sparse_index.item()] += 1
        if (j + 1) % 100 == 0:
            huffman_end_time = time.time()
            huffman_time_duration = (huffman_end_time - start_time)
            print(f"huffman {j - 99}-{j} test batches takes {huffman_time_duration} seconds", flush=True)
            start_time = time.time()

    print(f"program takes {time.time() - program_start_time} seconds", flush=True)

    with open(args.feature_frequency_mappings_path, 'wb') as file:
        pickle.dump(feature_frequency_mappings, file)
    print(f"Successfully dump feature_frequency_mappings dictionary to {args.feature_frequency_mappings_path}")


if __name__ == "__main__":
    run()