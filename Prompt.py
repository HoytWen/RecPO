import os
import random

random.seed(1958)


class Prompt:
    def __init__(self, prompt_path) -> None:
        assert os.path.isfile(prompt_path), "Please specify a prompt template"
        with open(prompt_path, 'r') as f:
            raw_prompts = f.read().splitlines()
        self.templates = [p.strip() for p in raw_prompts]

        self.historyList = []
        self.historyRatingList = []
        self.itemList = []
        self.trueSelection = ""

    @property
    def history_to_list(self):
        comb_list = []
        for i in range(len(self.historyList)):
            item_title = self.historyList[i]
            item_title = "Title: " + f"'{item_title}'"
            item_rating = "Rating: " + self.historyRatingList[i]
            item_comb = ", ".join([item_title, item_rating])
            comb_list.append(item_comb)

        return comb_list

    def __str__(self) -> str:
        # a = self.historyList
        prompt = self.templates[random.randint(0, len(self.templates) - 1)]

        # history = ", ".join(self.historyList)
        # history_ratings = ", ".join(self.historyRatingList)
        history = " | ".join(self.history_to_list)
        cans = ", ".join(self.itemList)

        # prompt = prompt.replace("[ViewHistory]", history)
        # prompt = prompt.replace("[RatingHistory]", history_ratings)
        prompt = prompt.replace("[HistoryAndRatings]", history)
        prompt = prompt.replace("[CansHere]", cans)
        prompt += " "

        return prompt
