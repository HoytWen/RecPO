import torch
import os.path as op
import numpy as np
import pickle as pkl
import torch.utils.data as data
from torch.utils.data import DataLoader

import pandas as pd
import random
from typing import List
import json


def decay_fn(rating, step, decay_factor):

    rating = np.array(rating)
    step = np.array(step)
    score = rating/(1+step)**decay_factor

    return np.round(score, 4)

class MovielensData_rating(data.Dataset):
    def __init__(self, data_dir=r'data/ref/movielens',
                 stage=None,
                 cans_num=10,
                 sep="::",
                 add_mv_year=False,
                 max_size=10000,
                 no_augment=True,
                 seq_len_min=5,
                 seq_len_max=20,
                 rating_threshold=4,
                 decay_factor = 0.5):
        self.__dict__.update(locals())
        self.data_dir = data_dir
        self.stage = stage
        self.add_mv_year = add_mv_year
        self.sep = sep
        self.aug = (stage == 'train') and not no_augment
        self.padding_item_id = 0
        self.padding_rating = 0
        self.check_files()
        self.cans_num = cans_num
        self.max_size = max_size
        self.seq_len_min = seq_len_min
        self.seq_len_max = seq_len_max
        self.rating_threshold = rating_threshold
        self.decay_factor = decay_factor


    def __len__(self):
        return len(self.session_data['seq'])

    def __getitem__(self, i):

        temp = self.session_data.iloc[i]

        if self.stage in ['train']:
            random_cans_num = max(0, self.cans_num-len(temp['follow_seq']))
            cans_id = self.negative_sampling(temp['seq_unpad'], temp['follow_seq'], random_cans_num)

            ranking_cans_id, ranking_cans_score = self.candidate_ranking(cans_id, temp['follow_rating'], random_cans_num)
            ranking_cans_name = [self.item_id2name[can] for can in ranking_cans_id]

            sample = {
                'seq': temp['seq_unpad'],
                'seq_name': temp['seq_title'],
                'seq_rating': temp['seq_rating_unpad'],
                'len_seq': len(temp['seq_unpad']),
                'seq_str': self.sep.join(temp['seq_title']),
                'follow_name': temp['follow_title'],
                'follow_rating': temp['follow_rating'],
                'cans': ranking_cans_id,
                'cans_name': ranking_cans_name,
                'cans_score': ranking_cans_score,
                'len_cans': self.cans_num,
                'user_id': temp['user_id'],
                'next_id': temp['next_item_id'],
                'next_title': temp['next_item_title'],
                'next_rating': temp['next_item_rating'],
                'correct_answer': ranking_cans_name[0],
                'item_score': ranking_cans_score[0]
            }

        else:
            random_cans_num = self.cans_num - 1
            cans_id = self.negative_sampling(temp['seq_unpad'], temp['next_item_id'], random_cans_num)
            cans_name = [self.item_id2name[can] for can in cans_id]

            sample = {
                'seq': temp['seq_unpad'],
                'seq_name': temp['seq_title'],
                'seq_rating': temp['seq_rating_unpad'],
                'len_seq': len(temp['seq_unpad']),
                'seq_str': self.sep.join(temp['seq_title']),
                'cans': cans_id,
                'cans_name': cans_name,
                'len_cans': self.cans_num,
                'user_id': temp['user_id'],
                'next_id': temp['next_item_id'],
                'next_rating': temp['next_item_rating'],
                'correct_answer': temp['next_item_title'],
            }

        return sample

    def candidate_ranking(self, cans_list, follow_ratings, sample_sum):

        if sample_sum == 0:
            rating_list = follow_ratings[:self.cans_num]
            step = np.arange(len(rating_list)).tolist()
        else:
            rating_list = follow_ratings + [3]*sample_sum
            step = np.arange(len(follow_ratings)).tolist() + [10]*sample_sum

        scores = decay_fn(rating_list, step, self.decay_factor)
        sorted_index = np.argsort(scores)[::-1]

        cans_list = [cans_list[x] for x in sorted_index]

        return cans_list, scores[sorted_index].tolist()

    def negative_sampling(self, seq_unpad, follow_seq, sample_num):

        if sample_num == 0:
            return follow_seq[:self.cans_num]

        follow_seq  = follow_seq if isinstance(follow_seq, List) else [follow_seq]
        canset = [i for i in list(self.item_id2name.keys()) if i not in seq_unpad and i not in follow_seq]
        candidates = follow_seq + random.sample(canset, sample_num)
        # random.shuffle(candidates)

        return candidates

    def check_files(self):
        self.item_id2name = self.get_movie_id2name()
        if self.stage == 'train':
            filename = "train_data_fseq.df"
        elif self.stage == 'val':
            filename = "Val_data.df"
        elif self.stage == 'test':
            filename = "Test_data.df"
        else:
            raise ValueError
        data_path = op.join(self.data_dir, filename)
        self.session_data = self.session_data4frame(data_path, self.item_id2name)

    def get_mv_title(self, s):
        sub_list = [", The", ", A", ", An"]
        for sub_s in sub_list:
            if sub_s in s:
                return sub_s[2:] + " " + s.replace(sub_s, "")
        return s

    def get_movie_id2name(self):

        movie_token2name = dict()
        item_path = op.join(self.data_dir, 'u.item')
        with open(item_path, 'r', encoding="ISO-8859-1") as f:
            for l in f.readlines()[1:]:
                ll = l.strip('\n').split('\t')
                title = self.get_mv_title(ll[1])

                if self.add_mv_year:
                    title = title + f' ({ll[2]})'

                movie_token2name[int(ll[0])] = title

        movie_id2name = dict()
        token2id_path = op.join(self.data_dir, 'token2id.json')
        with open(token2id_path, "r") as f:
            token2id = json.load(f)
            for token, idx in token2id.items():
                if token == '[PAD]':
                    continue
                title = movie_token2name[int(token)]
                movie_id2name[idx] = title

        return movie_id2name

    def session_data4frame(self, datapath, movie_id2name):
        train_data = pd.read_pickle(datapath)
        # train_data = train_data[train_data['len_seq'] >= self.seq_len_min]
        if self.stage in ['val', 'test']:
            train_data = train_data[train_data['next_item_rating']>=self.rating_threshold]

        else:
            train_data = train_data[train_data['len_seq'] >= self.seq_len_min]
            if len(train_data) > self.max_size:
                train_data = train_data.iloc[torch.randperm(len(train_data))[:self.max_size]]

        def remove_padding(xx):
            x = xx[:]
            for i in range(50):
                try:
                    x.remove(self.padding_item_id)
                except:
                    break
            return x[-self.seq_len_max:]

        train_data['seq_unpad'] = train_data['seq'].apply(remove_padding)
        train_data['seq_rating_unpad'] = train_data['seq_rating'].apply(remove_padding)

        def seq_to_title(x):
            return [movie_id2name[x_i] for x_i in x]
        train_data['seq_title'] = train_data['seq_unpad'].apply(seq_to_title)

        if self.stage in ['train']:
            train_data['follow_title'] = train_data['follow_seq'].apply(seq_to_title)

        def seq_to_rating(x):
            return [str(x_i) for x_i in x]
        train_data['seq_rating_unpad'] = train_data['seq_rating_unpad'].apply(seq_to_rating)

        def next_item_title(x):
            return movie_id2name[x]

        def next_item_rating(x):
            return str(x)

        train_data['next_item_title'] = train_data['next_item_id'].apply(next_item_title)
        train_data['next_item_rating'] = train_data['next_item_rating'].apply(next_item_rating)

        # def get_id_from_tumple(x):
        #     return x[0]
        #
        # def get_id_from_list(x):
        #     return [i[0] for i in x]

        # train_data['next'] = train_data['next'].apply(get_id_from_tumple)
        # train_data['seq'] = train_data['seq'].apply(get_id_from_list)
        # train_data['seq_unpad'] = train_data['seq_unpad'].apply(get_id_from_list)


        return train_data
    

if __name__ == "__main__":

    # train_dataloader = DataLoader(LastfmData(stage='train'), batch_size=2, shuffle=True)
    # test_dataloader = DataLoader(LastfmData(stage='test'), batch_size=8, shuffle=False)

    data_dir = '/home/ericwen/seq_rec_data/movielens-1m'
    data = MovielensData_rating(data_dir=data_dir, stage='val', cans_num=20)

    print(data[1])