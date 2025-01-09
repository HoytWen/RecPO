import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
from datasets import load_dataset
from transformers import LlamaTokenizer, LlamaForCausalLM, BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
from Prompt import *
import torch
from evaluate_batch import evaluate
from peft import PeftModel, prepare_model_for_kbit_training
from accelerate import Accelerator
import fire


def inference(dataset="",
              model_name="meta-llama/Llama-3.1-8B",
              prompt_path="./prompt/movie_rating2.txt",
              batch_size: int = 32,
              resume_from_checkpoint: str = "./output/Base-8B-RecDPO-wSFT-gpu4/",
              ):

    if "Base" in resume_from_checkpoint:
        assert "Instruct" not in model_name

    compute_dtype = getattr(torch, "bfloat16")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
        # load_in_8bit=True,
    )
    device_index = Accelerator().process_index
    device_map = {"": device_index}

    print(f"Load the base model {model_name}!")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        # quantization_config=bnb_config,
    )

    if resume_from_checkpoint:
        print(f"Evaluate the model checkpoint {resume_from_checkpoint}!")
        model = PeftModel.from_pretrained(model, resume_from_checkpoint)
    else:
        print(f"Use base model {model_name} for evaluation!")
    model.eval()

    if "Llama-3" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        tokenizer = LlamaTokenizer.from_pretrained(model_name)

    tokenizer.pad_token_id = (0)
    tokenizer.padding_side = "left"

    def convert_dict_to_prompt(d: dict):
        t = Prompt(prompt_path)
        d["historyList"] = d["historyList"].split("::") if isinstance(d["historyList"], str) else d["historyList"]
        t.historyList = d["historyList"]
        t.historyRatingList = d["historyRatingList"]
        t.itemList = d["itemList"]
        t.trueSelection = d["trueSelection"]
        return t

    def generate_and_tokenize_prompt(data_point):
        t = convert_dict_to_prompt(data_point)
        prompt = str(t)
        dic = data_point
        dic["prompt"] = prompt[:-1]
        # dic["prompt"] = prompt
        return dic

    data_files = {
        # "train": f"./data/movielens-1m/movielens-size10000-cans20-train.json",
        "test": "./data/movielens-1m/movielens-cans20-test-new.json",
    }

    data = load_dataset("json", data_files=data_files)
    data.cleanup_cache_files()
    print(data)

    test_data = data["test"].map(generate_and_tokenize_prompt)

    accuracy, valid_ratio = evaluate(model, tokenizer, test_data, batch_size=batch_size)
    print(accuracy, valid_ratio)

    print(f"The results is based on checkpoint {resume_from_checkpoint}!")

if __name__ == "__main__":
    fire.Fire(inference)
