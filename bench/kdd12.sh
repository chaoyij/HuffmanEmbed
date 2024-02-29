#!/bin/bash

#WARNING: must have compiled PyTorch

#check if extra argument is passed to the test
# if [[ $# == 1 ]]; then
#     dlrm_extra_option=$1
# else
#     dlrm_extra_option=""
# fi
# #echo $dlrm_extra_option

cuda_index=$1
echo "cuda_index: $cuda_index"

dlrm_extra_option=$2
echo "dlrm_extra_option: $dlrm_extra_option"

log_file_path=$3
echo "log_file_path: $log_file_path"

dlrm_generate_feature_frequency_mappings="python3 generate_feature_frequency_mappings.py"
dlrm_generate_huffman_coding="python3 generate_huffman_coding_tensors.py"
dlrm_pt_bin="python3 dlrm_s_pytorch.py"

# CUDA_VISIBLE_DEVICES=7 CUDA_LAUNCH_BLOCKING=1 \
# python3 dlrm_s_pytorch.py \
# --use-gpu \
# --arch-sparse-feature-size=64 \
# --arch-mlp-bot="13-512-256-64-64" \
# --arch-mlp-top="512-256-1" \
# --data-generation=dataset \
# --data-set=kdd12 \
# --loss-function=bce \
# --round-targets=True \
# --learning-rate=0.1 \
# --mini-batch-size=128 \
# --print-freq=1024 \
# --test-freq=1024 \
# --print-time \
# --test-mini-batch-size=16384 \
# --test-num-workers=16 \
# --cat-path="./input/kdd12_cat.bin" \
# --dense-path="" \
# --label-path="./input/kdd12_label.bin" \
# --count-path="./input/kdd12_count.bin" \
# $dlrm_extra_option 2>&1 | tee kdd12.log


# CUDA_VISIBLE_DEVICES=7 taskset -c 0-127 $dlrm_generate_feature_frequency_mappings --feature-frequency-mappings-path="huffman_coding_feature_frequency_mappings_kdd12_01252024.pkl" --data-generation=dataset --data-set=kdd12 --mini-batch-size=128 --test-num-workers=16 --test-mini-batch-size=16384 --cat-path="./input/kdd12_cat.bin" --dense-path="" --label-path="./input/kdd12_label.bin" --count-path="./input/kdd12_count.bin"
# CUDA_VISIBLE_DEVICES=6 taskset -c 0-127 $dlrm_generate_huffman_coding --feature-frequency-mappings-path="huffman_coding_feature_frequency_mappings_kdd12_01252024.pkl" --huffman-coding-tensors-path="huffman_coding_tensors_kdd12_01252024.pkl"

# Full dataset
# CUDA_VISIBLE_DEVICES=7 taskset -c 0-127 $dlrm_pt_bin --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kdd12 --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=128 --print-freq=1024 --print-time --test-freq=1024 --test-mini-batch-size=16384 --test-num-workers=16 --nepochs=1 --use-gpu --cat-path="./input/kdd12_cat.bin" --dense-path="" --label-path="./input/kdd12_label.bin" --count-path="./input/kdd12_count.bin" 
# Huffman Coding
CUDA_VISIBLE_DEVICES=$cuda_index \
python3 dlrm_s_pytorch.py \
--arch-sparse-feature-size=16 \
--arch-mlp-bot="13-512-256-64-16" \
--arch-mlp-top="512-256-1" \
--data-generation=dataset \
--data-set=kdd12 \
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
--use-gpu \
--cat-path="./input/kdd12_cat.bin" \
--dense-path="" \
--label-path="./input/kdd12_label.bin" \
--count-path="./input/kdd12_count.bin" \
$dlrm_extra_option 2>&1 | tee $log_file_path

echo "done"


# --sketch-flag \
# --notinsert-test \
# --compress-rate=0.005 \
# --hash-rate=0.8 \
# --sketch-threshold=500 \