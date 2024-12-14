import pandas as pd
import torch

from movielens_data import MovielensData_rating
import json
from tqdm import tqdm


if __name__ == "__main__":
    # splits = ["train", "val", "test"]
    splits = ['train']
    cans_num = 20
    data_dir = "/home/ericwen/seq_rec_data/movielens-1m"
    max_size = 50000
    for split in splits:
        data_split = MovielensData_rating(data_dir=data_dir, stage=split, cans_num=cans_num, max_size=max_size)
        dic_lis = []
        for i in tqdm(range(len(data_split))):
        ### Add random Shuffle
            cans_shuffle = torch.randperm(cans_num)
            if split in ['train']:
                dic = {
                    "historyList": data_split[i]["seq_name"],
                    "historyRatingList": data_split[i]["seq_rating"],
                    "itemList": [data_split[i]["cans_name"][x] for x in cans_shuffle],
                    "itemScoreList": [data_split[i]["cans_score"][x] for x in cans_shuffle],
                    "trueSelection": data_split[i]["correct_answer"],
                    'selectionScore': data_split[i]["item_score"],
                }
            else:
                dic = {
                    "historyList": data_split[i]["seq_name"],
                    "historyRatingList": data_split[i]["seq_rating"],
                    "itemList": [data_split[i]["cans_name"][x] for x in cans_shuffle],
                    "itemScoreList": [],
                    "trueSelection": data_split[i]["correct_answer"],
                    'selectionScore': data_split[i]["next_rating"],
                }
            dic_lis.append(dic)
        with open(f"movielens-1m/movielens-size{max_size}-cans{cans_num}-{split}.json", "w") as f:
            json.dump(dic_lis, f, indent=4)

