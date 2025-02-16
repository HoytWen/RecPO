import os
import random

random.seed(1958)


class Prompt:
    def __init__(self, prompt_path) -> None:
        assert os.path.isfile(prompt_path), "Please specify a prompt template"
        with open(prompt_path, 'r') as f:
            raw_prompts = f.read().splitlines()
        self.templates = [p.strip() for p in raw_prompts]
        self.rating_mode = True if "rating" in prompt_path else False
        self.historyList = []
        self.historyRatingList = []
        self.itemList = []
        self.trueSelection = ""

    @property
    def history_to_list(self):
        comb_list = []
        for i in range(len(self.historyList)):
            item_title = self.historyList[i]
            item_rating = self.historyRatingList[i]

            if self.rating_mode:
                item_rating = "Rating: " + item_rating
                item_comb = " | ".join([item_title, item_rating])
            else:
                item_comb = item_title
            comb_list.append(item_comb)

        return comb_list

    def __str__(self) -> str:
        # a = self.historyList
        prompt = self.templates[random.randint(0, len(self.templates) - 1)]

        # history = " | ".join(self.history_to_list)
        # cans = " | ".join(self.itemList)
        history = "\n ".join(self.history_to_list)
        cans = "\n ".join(self.itemList)

        if self.rating_mode:
            prompt = prompt.replace("[HistoryAndRatings]", history)
        else:
            prompt = prompt.replace("[History]", history)
        prompt = prompt.replace("[CansHere]", cans)
        prompt += " "

        return prompt


class BeerPrompt:
    def __init__(self, prompt_path) -> None:
        assert os.path.isfile(prompt_path), "Please specify a prompt template"
        with open(prompt_path, 'r') as f:
            raw_prompts = f.read().splitlines()
        self.templates = [p.strip() for p in raw_prompts]

        self.historyList = []
        self.historyRatingList = []
        self.itemList = []
        self.trueSelection = ""
        self.prompt_mode = prompt_path.split('.txt')[0][-1]

    @property
    def history_to_list(self):
        comb_list = []
        for i in range(len(self.historyList)):
            item_title = self.historyList[i]
            item_rating = self.historyRatingList[i].split('-')

            item_rating = f"Rating: {item_rating[4]}"
            # if self.prompt_mode == '1':
            # elif self.prompt_mode == '2':
            #     item_rating = f"Ap={item_rating[0]}, Ar={item_rating[1]}, P={item_rating[2]}, T={item_rating[3]}, O={item_rating[4]}"
            # else:
            #     item_rating = f"{item_rating[0]}, {item_rating[1]}, {item_rating[2]}, {item_rating[3]}, {item_rating[4]}"
            #
            item_comb = " | ".join([item_title, item_rating])
            comb_list.append(item_comb)

        return comb_list

    def __str__(self) -> str:
        # a = self.historyList
        prompt = self.templates[random.randint(0, len(self.templates) - 1)]

        history = "\n ".join(self.history_to_list)
        cans = "\n ".join(self.itemList)

        prompt = prompt.replace("[HistoryAndRatings]", history)
        prompt = prompt.replace("[CansHere]", cans)
        prompt += " "

        return prompt