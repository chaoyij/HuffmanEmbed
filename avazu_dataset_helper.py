import os
import os.path as osp
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import torch


default_data_path = osp.join(
    osp.split(osp.abspath(__file__))[0], './input')
default_criteo_path = osp.join(default_data_path, 'criteo')
default_avazu_path = osp.join(default_data_path, 'avazu')
default_company_path = osp.join(default_data_path, 'tencent')

class CTRDataset(object):
    def __init__(self, path):
        self.path = path
        self.raw_path = self.join('train.csv')
        self.phases = ['train', 'val', 'test']
        self.keys = ['dense', 'sparse', 'label']
        self.dtypes = [np.float32, np.int32, np.int32]
        self.shapes = [(-1, self.num_dense), (-1, self.num_sparse), (-1, 1)]

    @property
    def num_dense(self):
        raise NotImplementedError

    @property
    def num_sparse(self):
        raise NotImplementedError

    @property
    def num_embed(self):
        raise NotImplementedError

    @property
    def num_embed_separate(self):
        raise NotImplementedError

    def all_exists(self, paths):
        return all([osp.exists(pa) for pa in paths])

    def join(self, fpath):
        return osp.join(self.path, fpath)

    def download(self, path):
        raise NotImplementedError

    def read_from_raw(self, nrows=-1):
        raise NotImplementedError

    def read_csv(self, nrows=-1):
        path = self.raw_path
        print(f"path:{path}")
        if not osp.exists(path):
            self.download(path)
        if nrows > 0:
            df = pd.read_csv(path, nrows=nrows)
        else:
            df = pd.read_csv(path)
        return df

    def process_dense_feats(self, data, feats, inplace=True):
        if inplace:
            d = data
        else:
            d = data.copy()
        d = d[feats].fillna(0.0)
        for f in feats:
            d[f] = d[f].apply(lambda x: np.log(
                x+1) if x > 0 else 0)  # for criteo
        return d

    def process_sparse_feats(self, data, feats, inplace=True):
        from sklearn.preprocessing import LabelEncoder
        # process to embeddings.
        if inplace:
            d = data
        else:
            d = data.copy()
        d = d[feats].fillna("0")
        for f in feats:
            label_encoder = LabelEncoder()
            d[f] = label_encoder.fit_transform(d[f])
        feature_cnt = 0
        counts = [0]
        for f in feats:
            d[f] += feature_cnt
            feature_cnt += d[f].nunique()
            counts.append(feature_cnt)
        return d, counts

    def get_separate_fields(self, path, sparse, num_embed_fields):
        if osp.exists(path):
            return np.memmap(path, mode='r', dtype=np.int32).reshape(-1, self.num_sparse)
        else:
            accum = 0
            sparse = np.array(sparse)
            for i in range(self.num_sparse):
                sparse[:, i] -= accum
                accum += num_embed_fields[i]
            sparse.tofile(path)
            return sparse

    def accum_to_count(self, accum):
        count = []
        for i in range(len(accum) - 1):
            count.append(accum[i+1] - accum[i])
        return count

    def count_to_accum(self, count):
        accum = [0]
        for c in count:
            accum.append(accum[-1] + c)
        return accum

    def get_split_indices(self, num_samples):
        raise NotImplementedError

    def _binary_search_frequency(self, left, right, counter, target):
        if left >= right - 1:
            return left
        middle = (left + right) // 2
        high_freq_num = np.sum(counter >= middle)
        if high_freq_num > target:
            return self._binary_search_frequency(middle, right, counter, target)
        elif high_freq_num < target:
            return self._binary_search_frequency(left, middle, counter, target)
        else:
            return middle

    def get_frequency_counter(self, train_data, num_embed, fpath):
        if osp.exists(fpath):
            counter = np.fromfile(fpath, dtype=np.int32)
        else:
            counter = np.zeros((num_embed,), dtype=np.int32)
            for idx in tqdm(train_data.reshape(-1)):
                counter[idx] += 1
            counter.tofile(fpath)
        return counter

    def get_single_frequency_split(self, train_data, num_embed, top_percent, fpath, exact_split=False):
        fpath_parts = fpath.split('.')
        suffix = '_exact' if exact_split else ''
        real_fpath = '.'.join(
            fpath_parts[:-1]) + '_' + str(top_percent) + suffix + '.' + fpath_parts[-1]
        if osp.exists(real_fpath):
            result = np.fromfile(real_fpath, dtype=np.int32)
        else:
            dirpath, filepath = osp.split(fpath)
            cache_fpath = osp.join(dirpath, 'counter_' + filepath)
            counter = self.get_frequency_counter(
                train_data, num_embed, cache_fpath)
            nhigh = len(counter) * top_percent
            kth = self._binary_search_frequency(
                np.min(counter), np.max(counter) + 1, counter, nhigh)
            print(f'The threshold for high frequency is {kth}.')
            result = (counter >= kth).astype(np.int32)
            if exact_split:
                nhigh = round(nhigh)
                cur_nhigh = result.sum()
                if cur_nhigh < nhigh:
                    further_result = (counter == kth - 1).nonzero()[0]
                    further_result = np.random.choice(
                        further_result, size=nhigh - cur_nhigh, replace=False)
                    result[further_result] = 1
                    assert result.sum() == nhigh
                elif cur_nhigh > nhigh:
                    further_remove = (counter == kth).nonzero()[0]
                    further_remove = np.random.choice(
                        further_remove, size=cur_nhigh - nhigh, replace=False)
                    result[further_remove] = 0
                    assert result.sum() == nhigh
            print(
                f'Real ratio of high frequency is {np.sum(result) / len(result)}.')
            result.tofile(real_fpath)
        return result

    def get_whole_frequency_split(self, train_data, top_percent, exact_split=False):
        # now the filename is not correlated to top percent;
        # if modify top percent, MUST modify the fpath!
        freq_path = self.join('freq_split.bin')
        result = self.get_single_frequency_split(
            train_data, self.num_embed, top_percent, freq_path, exact_split)
        return result

    def get_separate_frequency_split(self, train_data, top_percent, exact_split=False):
        # now the filename is not correlated to top percent;
        # if modify top percent, MUST modify the fpath!
        separate_dir = self.join('freq_split_separate')
        os.makedirs(separate_dir, exist_ok=True)
        freq_paths = [
            osp.join(separate_dir, f'fields{i}.bin') for i in range(self.num_sparse)]
        results = [self.get_single_frequency_split(data, nemb, top_percent, fp, exact_split) for data, nemb, fp in zip(
            train_data, self.num_embed_separate, freq_paths)]
        return results

    def remap_split_frequency(self, frequency):
        hidx = 0
        lidx = 1
        remap_indices = np.zeros(frequency.shape, dtype=np.int32)
        for i, ind in enumerate(frequency):
            if ind:
                remap_indices[i] = hidx
                hidx += 1
            else:
                remap_indices[i] = -lidx
                lidx += 1
        return remap_indices

    def get_whole_remap(self, train_data, top_percent, exact_split=False):
        # now the filename is not correlated to top percent;
        # if modify top percent, MUST modify the fpath!
        suffix = '_exact' if exact_split else ''
        remap_path = self.join(f'freq_remap{top_percent}{suffix}.bin')
        if osp.exists(remap_path):
            remap_indices = np.fromfile(remap_path, dtype=np.int32)
        else:
            result = self.get_whole_frequency_split(
                train_data, top_percent, exact_split)
            remap_indices = self.remap_split_frequency(result)
            remap_indices.tofile(remap_path)
        return remap_indices

    def get_separate_remap(self, train_data, top_percent, exact_split=False):
        # now the filename is not correlated to top percent;
        # if modify top percent, MUST modify the fpath!
        separate_dir = self.join('freq_split_separate')
        os.makedirs(separate_dir, exist_ok=True)
        suffix = '_exact' if exact_split else ''
        remap_path = [
            osp.join(separate_dir, f'remap_fields{i}_{top_percent}{suffix}.bin') for i in range(self.num_sparse)]
        if self.all_exists(remap_path):
            remap_indices = [np.fromfile(rp, dtype=np.int32)
                             for rp in remap_path]
        else:
            results = self.get_separate_frequency_split(
                train_data, top_percent, exact_split)
            remap_indices = [self.remap_split_frequency(
                res) for res in results]
            for ind, rp in zip(remap_indices, remap_path):
                ind.tofile(rp)
        return remap_indices

    def get_whole_frequency_grouping(self, train_data, nsplit):
        cache_path = self.join(f'freq_grouping_{nsplit}.bin')
        if osp.exists(cache_path):
            grouping = np.fromfile(cache_path, dtype=np.int32)
        else:
            counter_path = self.join('counter_freq_split.bin')
            counter = self.get_frequency_counter(
                train_data, self.num_embed, counter_path)
            indices = np.argsort(counter)
            nignore = 0
            group_index = 0
            grouping = np.zeros(counter.shape, dtype=np.int32)
            cur_nsplit = nsplit
            while cur_nsplit != 0:
                threshold = (self.num_embed - nignore) / cur_nsplit
                assert int(threshold) > 0
                minvalue = counter[indices[nignore]]
                places = (counter == minvalue)
                cnt = places.sum()
                if cnt >= threshold:
                    nignore += cnt
                    grouping[places] = group_index
                    assert cur_nsplit > 1
                else:
                    cur_index = nignore + int(threshold) - 1
                    target_value = counter[indices[cur_index]]
                    assert target_value > minvalue
                    offset = 1
                    while True:
                        if cur_index + offset >= self.num_embed or counter[indices[cur_index + offset]] != target_value:
                            ending = cur_index + offset
                            break
                        elif cur_index - offset <= nignore or counter[indices[cur_index - offset]] != target_value:
                            ending = cur_index - offset + 1
                            break
                        offset += 1
                    assert ending > nignore
                    grouping[indices[nignore:ending]] = group_index
                    nignore = ending
                cur_nsplit -= 1
                group_index += 1
            if nignore < self.num_embed:
                grouping[indices[nignore:self.num_embed]] = group_index - 1
            grouping.tofile(cache_path)
        return grouping

    def process_all_data(self, separate_fields=False):
        return self.process_all_data_by_day(separate_fields)

    def process_all_data_by_day(self, separate_fields=False):
        all_data_path = [
            [self.join(f'kaggle_processed_{ph}_{k}.bin') for k in self.keys] for ph in self.phases]

        data_ready = self.all_exists(sum(all_data_path, []))

        if not data_ready:
            ckeys = self.keys + ['count']
            cdtypes = self.dtypes + [np.int32]
            cshapes = self.shapes + [(-1,)]
            pro_paths = [
                self.join(f'kaggle_processed_{k}.bin') for k in ckeys]
            pro_data_ready = self.all_exists(pro_paths)

            if not pro_data_ready:
                print("Reading raw data={}".format(self.raw_path))
                all_data = self.read_from_raw()
                for data, path, dtype in zip(all_data, pro_paths, cdtypes):
                    data = np.array(data, dtype=dtype)
                    data.tofile(path)
            data_with_accum = [np.fromfile(path, dtype=dtype).reshape(
                shape) for path, dtype, shape in zip(pro_paths, cdtypes, cshapes)]
            accum = data_with_accum[-1]
            input_data = data_with_accum[:-1]
            counts = self.accum_to_count(accum)

            indices = self.get_split_indices(input_data[0].shape[0])

            # create training, validation, and test sets
            for ind, cur_paths in zip(indices, all_data_path):
                cur_data = [d[ind] for d in input_data]
                for d, p in zip(cur_data, cur_paths):
                    d.tofile(p)
        else:
            count = np.fromfile(
                self.join(f'kaggle_processed_count.bin'), dtype=np.int32)
            counts = self.accum_to_count(count)
        assert counts == self.num_embed_separate

        def get_data(phase):
            index = {'train': 0, 'val': 1, 'test': 2}[phase]
            sparse_index = self.keys.index('sparse')
            memmap_data = [np.memmap(p, mode='r', dtype=dtype).reshape(
                shape) for p, dtype, shape in zip(all_data_path[index], self.dtypes, self.shapes)]
            if separate_fields:
                memmap_data[sparse_index] = self.get_separate_fields(
                    self.join(f'kaggle_processed_{phase}_sparse_sep.bin'), memmap_data[sparse_index], counts)
            return memmap_data

        training_data = get_data('train')
        validation_data = get_data('val')
        testing_data = get_data('test')
        return tuple(zip(training_data, validation_data, testing_data))

    def check_skewness(self):
        sorted_all_counter_path = self.join('counter_all_sorted.bin')
        if not osp.exists(sorted_all_counter_path):
            all_counter_path = self.join('counter_all.bin')
            if not osp.exists(all_counter_path):
                counter = self.get_frequency_counter(
                    np.memmap(self.join(f'kaggle_processed_train_sparse.bin'), mode='r', dtype=np.int32), self.num_embed, self.join('counter_freq_split.bin'))
                for ph in ('val', 'test'):
                    cur_sparse = np.memmap(
                        self.join(f'kaggle_processed_{ph}_sparse.bin'), mode='r', dtype=np.int32)
                    for i in tqdm(cur_sparse.reshape(-1)):
                        counter[i] += 1
                counter.tofile(all_counter_path)
            else:
                counter = np.fromfile(all_counter_path)
            counter = np.sort(counter)
            counter.tofile(sorted_all_counter_path)

    def pruning_to_different_skewness(self, reserve_rate, alpha):
        assert 0 < reserve_rate < 1 and -1 <= alpha <= 1
        sum_file = self.join('sumfreq.bin')
        if not osp.exists(sum_file):
            all_counter = np.fromfile(
                self.join('counter_all.bin'), dtype=np.int32)
            sparse = np.memmap(self.join('kaggle_processed_sparse.bin'),
                               mode='r', dtype=np.int32).reshape(-1, self.num_sparse)
            scores = np.zeros(sparse.shape[0], dtype=np.int32)
            for i, sp in enumerate(tqdm(sparse)):
                scores[i] = all_counter[sp].sum()
            scores.tofile(sum_file)
        else:
            scores = np.fromfile(sum_file, dtype=np.int32)
        average = scores.mean()
        larger = scores.max() - average
        smaller = average - scores.min()
        if alpha > 0:
            larger = (1 - reserve_rate) / larger
            smaller = reserve_rate / smaller
            scale = min(larger, smaller)
        else:
            larger = reserve_rate / larger
            smaller = (1 - reserve_rate) / smaller
            scale = min(larger, smaller)
        rands = np.random.random_sample(size=scores.shape[0])
        rands = rands - scale * alpha * (scores - average)
        reserved = (rands < reserve_rate)
        reserved.tofile(self.join(f'reserving_{reserve_rate}_{alpha}.bin'))
        return reserved

    def pruning_to_different_skewness_greedy(self, target_num, alpha):
        assert 0 < target_num and alpha in (-1, 1)
        sum_file = self.join('sumfreq.bin')
        if not osp.exists(sum_file):
            all_counter = np.fromfile(
                self.join('counter_all.bin'), dtype=np.int32)
            sparse = np.memmap(self.join('kaggle_processed_sparse.bin'),
                               mode='r', dtype=np.int32).reshape(-1, self.num_sparse)
            scores = np.zeros(sparse.shape[0], dtype=np.int32)
            for i, sp in enumerate(tqdm(sparse)):
                scores[i] = all_counter[sp].sum()
            scores.tofile(sum_file)
        else:
            scores = np.fromfile(sum_file, dtype=np.int32)
        if alpha == 1:
            more_target = len(scores) - target_num
        else:
            more_target = target_num
        l, r = scores.min(), scores.max()
        while (r - l) > 0.5:
            mid = (l + r) / 2
            num_more = (scores > mid).sum()
            if num_more == more_target:
                break
            elif num_more > more_target:
                l = mid
            else:
                r = mid
        mid = int(mid)
        if alpha == 1:
            if (scores > mid).sum() > more_target:
                mid += 1
            reserved = (scores <= mid)
            reserved.tofile(self.join(f'reserving_{target_num}_{alpha}.bin'))
        else:
            if (scores > mid).sum() < more_target:
                mid -= 1
            reserved = (scores > mid)
            reserved.tofile(self.join(f'reserving_{target_num}_{alpha}.bin'))
        return reserved

    def masked_counter(self, mask_file, target_file):
        mask = np.fromfile(self.join(mask_file), dtype=np.bool_)
        sparse = np.memmap(self.join('kaggle_processed_sparse.bin'),
                           mode='r', dtype=np.int32).reshape(-1, self.num_sparse)
        sparse = sparse[mask]
        print('shape:', sparse.shape)
        counter = np.zeros((self.num_embed,), dtype=np.int32)
        for sp in tqdm(sparse):
            counter[sp] += 1
        counter.tofile(self.join(target_file))


class AvazuDataset(CTRDataset):
    def __init__(self, path=None):
        if path is None:
            path = default_avazu_path
        # please download manually from https://www.kaggle.com/c/avazu-ctr-prediction/data
        super().__init__(path)
        self.keys = ['sparse', 'label']
        self.dtypes = [np.int32, np.int32]
        self.shapes = [(-1, self.num_sparse), (-1, 1)]

    @property
    def num_dense(self):
        return 0

    @property
    def num_sparse(self):
        return 22

    @property
    def num_embed(self):
        return 9449445

    @property
    def num_embed_separate(self):
        return [240, 7, 7, 4737, 7745, 26, 8552, 559, 36,
                2686408, 6729486, 8251, 5, 4, 2626, 8, 9, 435, 4, 68, 172, 60]

    def read_from_raw(self, nrows=-1):
        df = self.read_csv(nrows)
        sparse_feats = [
            col for col in df.columns if col not in ('click', 'id')]
        assert len(sparse_feats) == self.num_sparse
        labels = df['click']
        sparse, counts = self.process_sparse_feats(df, sparse_feats)
        # return torch.tensor(sparse.values), torch.tensor(labels), torch.tensor(counts)
        return sparse, labels, counts

    def get_split_indices(self, num_samples):
        # get exactly number of samples of last day by 'hour' column
        n_last_day = 4218938
        indices = np.arange(num_samples)
        train_indices = indices[:-n_last_day]
        test_indices = indices[-n_last_day:]
        test_indices, val_indices = np.array_split(test_indices, 2)
        # randomize
        train_indices = np.random.permutation(train_indices)
        print("Randomized indices across days ...")
        indices = [train_indices, val_indices, test_indices]
        return indices


if __name__ == "__main__":
    avazu = AvazuDataset()
    X_cat, y, counts = avazu.read_from_raw()
    # print(f"X_cat type: {type(X_cat)} X_cat shape: {X_cat.shape}")
    # print(f"y type: {type(y)} y shape: {y.shape}")
    # print(f"counts type: {type(counts)} counts shape: {len(counts)}")
    # print("counts:", counts)
    cat_path = "./input/avazu_cat.bin"
    label_path = "./input/avazu_label.bin"
    count_path = "./input/avazu_count.bin"
    X_cat_array = X_cat.values.astype(np.int32)
    label_array = y.values.astype(np.int32)
    count_array = np.array(counts, dtype=np.int32)
    X_cat_array.tofile(cat_path)
    label_array.tofile(label_path)
    count_array.tofile(count_path)

    memmap_X_cat_array = np.memmap(cat_path, dtype=np.int32, mode='r+', shape=(40428967,22))
    memmap_label_array = np.memmap(label_path, dtype=np.int32, mode='r+', shape=(40428967,))
    memmap_count_array = np.memmap(count_path, dtype=np.int32, mode='r+', shape=(23,))

    # Access and modify the memory-mapped array
    print("Original memmap_X_cat_array:", memmap_X_cat_array)
    print("Original memmap_label_array:", memmap_label_array)
    print("Original memmap_count_array:", memmap_count_array)


