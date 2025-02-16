import pandas as pd
import torch

from movielens_data import MovielensData_rating
from amazonbooks_data import AmazonBookData_rating
from steam_data import SteamData_rating
from beeradvocate_data import BeerAdvocateData_rating
import json
from tqdm import tqdm


if __name__ == "__main__":
    ds_name = 'amazon-books'
    splits = ["train"]
    cans_num = 20
    data_dir = f'path/{ds_name}'
    max_size = 10000
    for split in splits:
        data_split = AmazonBookData_rating(data_dir=data_dir, stage=split, cans_num=cans_num, max_size=max_size, rating_threshold=4)
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
                    'selectionScore': data_sample["next_rating"][-1],
                    "ratingNegative": [],
                    "randomNegative": data_sample["random_negative_cans_name"],
                }
            dic_lis.append(dic)

        if split in ['train']:
            with open(f"{ds_name}/{ds_name}-size{max_size}-cans{cans_num}-{split}.json", "w") as f:
                json.dump(dic_lis, f, indent=4, ensure_ascii=False)
        else:
            with open(f"{ds_name}/{ds_name}-cans{cans_num}-{split}.json", "w") as f:
                json.dump(dic_lis, f, indent=4, ensure_ascii=False)

