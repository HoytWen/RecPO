import numpy as np
import torch

import transformers
from typing import List
from datasets import load_dataset
import json
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig, GenerationMixin
from peft import PeftModel
import torch.nn as nn
from torch.utils.data import DataLoader
import random
from fire import Fire
from tqdm import tqdm

device_map = "auto"


def evaluate(
        model,
        tokenizer,
        val_data,
        batch_size: int = 32,
        save_output: bool = False
):
    def output_generate(
            prompts,
            temperature=0,
    ):
        # print([len(prompt) for prompt in prompts])
        inputs = tokenizer(prompts, return_tensors="pt", truncation=True, padding=True, max_length=1024).to(
            model.device)
        generation_config = GenerationConfig(
            # temperature=0.3,
            # top_k=1,
            # do_sample=True,
            do_sample=False,
        )
        generation_output = model.generate(
            **inputs,
            pad_token_id=tokenizer.pad_token_id,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=128
        )
        s = generation_output.sequences
        output = tokenizer.batch_decode(s, skip_special_tokens=True)
        output = [_.strip() for _ in output]
        return output

    targets = []
    inputs = []
    cans = []
    history_rating_list = []
    target_scores = []
    dic_lis = []
    for elm in val_data:
        prompt = elm["prompt"]
        target = elm["trueSelection"]
        history_rating = [float(x.split('-')[-1]) for x in elm['historyRatingList']]
        target_score = elm["selectionScore"]
        targets.append(target)
        inputs.append(prompt)
        cans.append(elm["itemList"])
        target_scores.append(target_score)
        history_rating_list.append(history_rating)

    batch_num = (len(inputs) - 1) // batch_size + 1
    score = 0
    valid = 0
    correct_var_list = []
    valid_var_list = []
    all_var_list = []
    for i in tqdm(range(batch_num), desc="Testing..."):
        start = i * batch_size
        end = min(len(inputs), start + batch_size)
        batch_inputs = inputs[start:end]
        outputs = output_generate(batch_inputs)
        batch_targets = targets[start:end]
        batch_cans = cans[start:end]
        batch_target_scores = target_scores[start:end]
        batch_history_rating = history_rating_list[start:end]
        for input_text, output, target, candidates, target_score, history_rating in zip(batch_inputs, outputs,
                                                                                        batch_targets, batch_cans,
                                                                                        batch_target_scores, batch_history_rating):
            # selection = output[len(input_text):]
            selection = output.split(" Answer:")[-1]
            num_cans = sum([1 for can in candidates if can in selection])
            valid_flag = False
            correct_flag = False
            print(input_text)
            print(candidates)
            print(selection)
            # print(f"Target: {target}, Target Score: {target_score}")
            print(f"Target: {target}")
            if num_cans == 1:
                valid += 1
                valid_flag = True

                if target in selection:
                    score += 1
                    correct_var_list.append(np.var(history_rating))
                    correct_flag = True
                    print(f"Score increased to {score}")

                # if num_high_rating == 1:
                #     high_rating_ratio += 1
                #     print(f"High rating ratio increased to {high_rating_ratio}")

                valid_var_list.append(np.var(history_rating))
                print(f"Valid ratio increased to {valid}")

            all_var_list.append(np.var(history_rating))
            dic = {
                "prompt": input_text,
                "candidates": candidates,
                "selection": selection,
                "selectionScore": target_score,
                "target": target,
                "validFlag": valid_flag,
                "correctFlag": correct_flag,
            }
            dic_lis.append(dic)
            print("\n")


    if save_output:
        return (score / len(inputs),  valid / len(inputs),
                np.mean(correct_var_list), np.mean(valid_var_list), np.mean(all_var_list), dic_lis)
    else:
        return (score / len(inputs), valid / len(inputs),
                np.mean(correct_var_list), np.mean(valid_var_list), np.mean(all_var_list))
