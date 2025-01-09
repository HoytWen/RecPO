import pandas as pd
import torch

from movielens_data import MovielensData_rating
import json
from tqdm import tqdm


if __name__ == "__main__":
    splits = ["train"]
    # splits = ["val", "test", "train"]
    cans_num = 20
    data_dir = "/home/ericwen/seq_rec_data/movielens-1m"
    max_size = 10000
    for split in splits:
        data_split = MovielensData_rating(data_dir=data_dir, stage=split, cans_num=cans_num, max_size=max_size)
        dic_lis = []
        for i in tqdm(range(len(data_split))):
        ### Add random shuffle
            data_sample = data_split[i]
            cans_shuffle = torch.randperm(cans_num)
            if split in ['train']:
                dic = {
                    "historyList": data_sample["seq_name"],
                    "historyRatingList": data_sample["seq_rating"],
                    "itemList": [data_sample["cans_name"][x] for x in cans_shuffle],
                    "itemScoreList": [data_sample["cans_score"][x] for x in cans_shuffle],
                    "trueSelection": data_sample["correct_answer"],
                    "selectionScore": data_sample["item_score"],
                    "ratingNegative": data_sample["rating_negative_cans_name"],
                    "randomNegative": data_sample["random_negative_cans_name"],
                }
            else:
                dic = {
                    "historyList": data_sample["seq_name"],
                    "historyRatingList": data_sample["seq_rating"],
                    "itemList": [data_sample["cans_name"][x] for x in cans_shuffle],
                    "itemScoreList": [],
                    "trueSelection": data_sample["correct_answer"],
                    'selectionScore': data_sample["next_rating"],
                    "ratingNegative": [],
                    "randomNegative": data_sample["random_negative_cans_name"],
                }
            dic_lis.append(dic)

        if split in ['train']:
            with open(f"movielens-1m/movielens-size{max_size}-cans{cans_num}-{split}-new.json", "w") as f:
                json.dump(dic_lis, f, indent=4)
        else:
            with open(f"movielens-1m/movielens-cans{cans_num}-{split}-new.json", "w") as f:
                json.dump(dic_lis, f, indent=4)

