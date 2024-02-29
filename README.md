# HuffmanEmbed: Using Huffman Coding for Embedding Table Compression in Deep Learning Recommendation Models

## Scripts
This repository is submitted for VLDB 2024. Using the scripts will allow you to run HuffmanEmbed and all the included baselines, namely full dataset without any compression, Hash and CAFE for all datasets. The repository also supports WDL and DCN.

Our implementation builds upon DLRM repo: https://github.com/facebookresearch/dlrm, and CAFE repo: https://github.com/HugoZHL/CAFE

1. The code supports interface with the [Criteo Kaggle Display Advertising Challenge Dataset](https://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/).

   - The model can be trained using the following script

     - Follow the instruction on Facebook DLRM to generate kaggleAdDisplayChallenge_processed.npz
     - Set the parameters --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz in the script.

     ```
     ./bench/criteo_kaggle.sh
     ```

2. The code also supports another two datasets [Avazu](https://kaggle.com/competitions/avazu-ctr-prediction) and [KDD12](https://kaggle.com/competitions/kddcup2012-track2).
   - Please do the following to prepare the dataset for use with this code:
     - Set the parameters cat_path, dense_path, label_path and count_path in the script.

   - The model can be trained using the following script

     ```
     ./bench/avazu.sh
     ./bench/kdd12.sh
     ```

3. The code provides three models to train the dataset:
   - dlrm:

     ```
     ./bench/criteo_kaggle.sh
     ```
   - wdl:

     ```
     ./bench/wdl.sh
     ```
   - dcn:

     ```
     ./bench/dcn.sh
     ```

4. The code provides methods for generating baseline embedding layers:

   - Full embedding with the following script

     ```
     ./bench/criteo_kaggle.sh
     ```

   - Hash embedding with the following script

     ```
     ./bench/criteo_kaggle.sh "--hash-flag --compress-rate=0.001"
     ```

   - CAFE with the following script

     ```
     ./bench/criteo_kaggle.sh "--sketch-flag --compress-rate=0.001 --hash-rate=0.3"
     ```


## HuffmanEmbed Setup

1. Use generate_feature_frequency_mappings.py to generate feature frequency mappings:
  ```
  python3 generate_feature_frequency_mappings.py --feature-frequency-mappings-path "huffman_coding_feature_frequency_mappings_path.pkl" --data-generation=dataset --data-set=kaggle --raw-data-file=./input/train.txt --processed-data-file=./input/kaggleAdDisplayChallenge_processed.npz --mini-batch-size=128 --test-num-workers=16 --test-mini-batch-size=16384
  ```

### Parameters
| name | explanation |
|----------|----------|
| --feature-frequency-mappings-path(string) | output path of feature frequency mappings |


2. Use generate_huffman_coding_tensors.py to generate Huffman Coding Tensors:
  ```
  python3 generate_huffman_coding_tensors.py --feature-frequency-mappings-path="huffman_coding_feature_frequency_mappings_path.pkl" --huffman-coding-tensors-path="huffman_coding_tensors_path.pkl" --huffman-coding-digits=2 --use-reverse-and-shift > output_generate_huffman_coding_tensors.log 2>&1 &
  ```

### Parameters
| name | explanation |
|----------|----------|
| --feature-frequency-mappings-path(string) | input path of feature frequency mappings generated from 1 |
| --huffman-coding-tensors-path(string) | output path of huffman coding tensors |
| --huffman-coding-digits(int) | number of Huffman embedding tables (i.e. N-ary) |
| --use-reverse-and-shift(bool)| use "reverse and shift" decoding algorithm |

3. Use scripts in the bench directory to train models with HuffmanEmbed:
  ```
  ./bench/criteo_kaggle.sh "0" "--arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --use-huffman-coding --capacity=52 --num-huffman-digits=2 --huffman-coding-tensors-path="huffman_coding_tensors_path.pkl"" "dlrm_kaggle_huffman_coding_log.txt" > output.log 2>&1 &
  ```

### Parameters
| name | explanation |
|----------|----------|
| --use-huffman-coding(bool) | use HuffmaneEmbed method for embedding compression|
| --capacity(int) | minimum number of rows required for embedding compression|
| --num-huffman-digits(int) | number of Huffman embedding tables (i.e. N-ary) |
| --huffman-coding-tensors-path(string) | input path of huffman coding tensors generated from 2|


## Citation
If you find this work useful, please cite our paper "HuffmanEmbed: Using Huffman Coding for Embedding Table Compression in Deep Learning Recommendation Models".
