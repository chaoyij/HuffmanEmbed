#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#WARNING: must have compiled PyTorch

#check if extra argument is passed to the test
cuda_index=$1
echo "cuda_index: $cuda_index"

dlrm_extra_option=$2
echo "dlrm_extra_option: $dlrm_extra_option"

log_file_path=$3
echo "log_file_path: $log_file_path"

# if [[ $# == 1 ]]; then
#     dlrm_extra_option=$1
# else
#     dlrm_extra_option=""
# fi
# echo $dlrm_extra_option

dlrm_generate_feature_frequency_mappings="python3 generate_feature_frequency_mappings.py"
dlrm_generate_huffman_coding="python3 generate_huffman_coding_tensors.py"
dlrm_pt_bin="python3 dlrm_s_pytorch.py"

CUDA_VISIBLE_DEVICES=$cuda_index \
python3 dlrm_s_pytorch.py \
--use-gpu \
--arch-mlp-top="512-256-1" \
--data-generation=dataset \
--data-set=kaggle \
--raw-data-file=./input/train.txt \
--processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz \
--loss-function=bce \
--round-targets=True \
--learning-rate=0.1 \
--mini-batch-size=128 \
--print-freq=1024 \
--print-time \
--test-freq=40960 \
--test-mini-batch-size=16384 \
--test-num-workers=16 \
--nepochs=1 \
$dlrm_extra_option 2>&1 | tee $log_file_path

# --arch-sparse-feature-size=16 \
# --arch-mlp-bot="13-512-256-64-16" \
# CUDA_VISIBLE_DEVICES=7 taskset -c 0-127 $dlrm_generate_feature_frequency_mappings --feature-frequency-mappings-path "huffman_coding_feature_frequency_mappings_01252024.pkl" --data-generation=dataset --data-set=kaggle --raw-data-file=./input/train.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz --mini-batch-size=128 --test-num-workers=16 --test-mini-batch-size=16384
# CUDA_VISIBLE_DEVICES=7 taskset -c 0-127 $dlrm_generate_huffman_coding --feature-frequency-mappings-path "huffman_coding_feature_frequency_mappings_01252024.pkl" --huffman-coding-tensors-path "huffman_coding_tensors_01252024.pkl"
# CUDA_VISIBLE_DEVICES=6 taskset -c 0-127 $dlrm_pt_bin --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle --raw-data-file=./input/train.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=128 --print-freq=1024 --print-time --test-freq=1024 --test-mini-batch-size=16384 --test-num-workers=16 --nepochs=1 --use-gpu --use-huffman-coding --capacity 52

# CAFE
# CUDA_VISIBLE_DEVICES=0 taskset -c 0-127 $dlrm_pt_bin --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle --raw-data-file=./input/train.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=128 --print-freq=1024 --print-time --test-freq=1024 --test-mini-batch-size=16384 --test-num-workers=16 --nepochs=1 --use-gpu --sketch-flag --compress-rate=0.001 --hash-rate=0.3
# CUDA_VISIBLE_DEVICES=1 taskset -c 0-127 $dlrm_pt_bin --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle --raw-data-file=./input/train.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=128 --print-freq=1024 --print-time --test-freq=1024 --test-mini-batch-size=16384 --test-num-workers=16 --nepochs=1 --use-gpu --sketch-flag --compress-rate=0.1 --hash-rate=0.7 --sketch-threshold=300

# AdaEmbed
# CUDA_VISIBLE_DEVICES=0 taskset -c 0-127 $dlrm_pt_bin --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle --raw-data-file=./input/train.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=128 --print-freq=1024 --print-time --test-freq=1024 --test-mini-batch-size=16384 --test-num-workers=16 --nepochs=1 --use-gpu --ada-flag --compress-rate=0.5

# MD Embed
# CUDA_VISIBLE_DEVICES=0 taskset -c 0-127 $dlrm_pt_bin --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle --raw-data-file=./input/train.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=128 --print-freq=1024 --print-time --test-freq=1024 --test-mini-batch-size=16384 --test-num-workers=16 --nepochs=1 --use-gpu --md-flag --compress-rate=0.1

# QR Embed
# CUDA_VISIBLE_DEVICES=0 taskset -c 0-127 $dlrm_pt_bin --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle --raw-data-file=./input/train.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=128 --print-freq=1024 --print-time --test-freq=1024 --test-mini-batch-size=16384 --test-num-workers=16 --nepochs=1 --use-gpu --qr-flag --qr-collisions=10

# --hash-flag \
# --compress-rate=0.001 \

echo "done"

# --cat-path="../criteo_kaggle/kaggle_processed_sparse_sep.bin" \
# --dense-path="../criteo_kaggle/kaggle_processed_dense.bin" \
# --label-path="../criteo_kaggle/kaggle_processed_label.bin" \
# --count-path="../criteo_kaggle/kaggle_processed_count.bin" \