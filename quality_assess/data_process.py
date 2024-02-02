import json
import fire
import random

GLOBAL_PATH = "/home/jiang.yank/work/"
def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data
def main(
    datasets: list = None,
    #prompt_style: list = None,
    num_prompts: int=20,
    seed=0
):
    prompt_style = ['13b','brief_13b','brief+_13b']
    select_samples = [[],[],[]]
    for dataset in datasets:
        if dataset == "mmlu":
            for i in range(len(prompt_style)):
                random.seed(seed)
                path = f"{GLOBAL_PATH}llama_exp/inference_tests/prompt_sweeping/batched_inference/auto_annotate/{dataset}/samples_800_seed_0/{dataset}_5000_{prompt_style[i]}.json"
                data = read_json_file(path)
                samples = random.sample(data, num_prompts)
                for sample in samples:
                    select_samples[i].append(sample)
        elif dataset == "triviaqa":
            for i in range(len(prompt_style)):
                random.seed(seed)
                path = f"{GLOBAL_PATH}llama_exp/inference_tests/prompt_sweeping/batched_inference/auto_annotate/{dataset}/{dataset}_800_{prompt_style[i]}.json"
                data = read_json_file(path)
                samples = random.sample(data, num_prompts)
                for sample in samples:
                    select_samples[i].append(sample)
        elif dataset == "naturalqa":
            for i in range(len(prompt_style)):
                random.seed(seed)
                path = f"{GLOBAL_PATH}llama_exp/inference_tests/prompt_sweeping/batched_inference/auto_annotate/{dataset}/samples_800_seed_0/{dataset}_5000_{prompt_style[i]}.json"
                data = read_json_file(path)
                samples = random.sample(data, num_prompts)
                for sample in samples:
                    select_samples[i].append(sample)
        elif dataset == "helpful":
            for i in range(len(prompt_style)):
                random.seed(seed)
                path = f"{GLOBAL_PATH}llama_exp/inference_tests/prompt_sweeping/batched_inference/auto_annotate/{dataset}_base/{dataset}_all_{prompt_style[i]}.json"
                data = read_json_file(path)
                samples = random.sample(data, num_prompts)
                for sample in samples:
                    select_samples[i].append(sample)
    for i in range(len(prompt_style)):
        with open(f"./test_data/{prompt_style[i]}.json", "w") as f:
            json.dump(select_samples[i], f, indent=4)

   
                






if __name__ == "__main__":
    fire.Fire(main)