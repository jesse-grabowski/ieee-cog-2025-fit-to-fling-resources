import string

from openai import OpenAI
from string import Template
import os

ollama_client = OpenAI(
    api_key="ollama",  # dummy key
    base_url="http://localhost:11434/v1"
)

def load_prompt(path):
    with open(path, 'r') as file:
        return file.read()

def make_dirs(trial_name, letter):
    os.makedirs(f'../ablation_data/{trial_name}/raw/{letter}', exist_ok=True)

def run_baseline_trial(name, model, seed, prompts):
    loaded_prompts = [load_prompt(p) for p in prompts]
    for c in string.ascii_uppercase:
        make_dirs(name, c)
        for i in range(len(prompts)):
            run_baseline_inference(name, model, seed, loaded_prompts[i], c, i)

def run_baseline_inference(trial_name, ollama_model, seed, prompt, letter, number):
    template = Template(prompt)
    chat_completion = ollama_client.chat.completions.create(
        messages=[{"role": "user", "content": template.substitute(content=letter)}],
        model=ollama_model,
        temperature=1,
        seed=seed,
        n=1,
    )

    response = chat_completion.choices[0].message.content
    with open(f'../ablation_data/{trial_name}/raw/{letter}/{trial_name}_{letter}_{number + 1}.txt', 'w') as file:
        file.write(response)

def run_diversity_trial(name, model, seed, prompts):
    diversify = load_prompt(prompts[0])
    recall = load_prompt(prompts[1])
    decode = load_prompt(prompts[2])
    for c in string.ascii_uppercase:
        make_dirs(name, c)
        for i in range(64):
            run_diversity_inference(name, model, seed, diversify, recall, decode, c, i)

_inference_cache = {}
def run_diversity_inference(trial_name, ollama_model, seed, diversify, recall, decode, letter, number):
    if (0, letter, number) in _inference_cache:
        diversify_out = _inference_cache[(0, letter, number)]
    else:
        chat_completion = ollama_client.chat.completions.create(
            messages=[{"role": "user", "content": Template(diversify).substitute(letter=letter, number=number)}],
            model=ollama_model,
            temperature=1,
            seed=seed,
            n=1,
        )
        diversify_out = chat_completion.choices[0].message.content
        _inference_cache[(0, letter, number)] = diversify_out

    if (1, diversify_out) in _inference_cache:
        recall_out = _inference_cache[(1, diversify_out)]
    else:
        chat_completion = ollama_client.chat.completions.create(
            messages=[{"role": "user", "content": Template(recall).substitute(content=diversify_out)}],
            model=ollama_model,
            temperature=1,
            seed=seed,
            n=1,
        )
        recall_out = chat_completion.choices[0].message.content
        _inference_cache[(1, diversify_out)] = recall_out

    if (2, recall_out) in _inference_cache:
        decode_out = _inference_cache[(2, recall_out)]
    else:
        chat_completion = ollama_client.chat.completions.create(
            messages=[{"role": "user", "content": Template(decode).substitute(content=recall_out)}],
            model=ollama_model,
            temperature=1,
            seed=seed,
            n=1,
        )
        decode_out = chat_completion.choices[0].message.content
        _inference_cache[(2, recall_out)] = decode_out


    with open(f'../ablation_data/{trial_name}/raw/{letter}/{trial_name}_{letter}_{number + 1}.txt', 'w') as file:
        file.write(decode_out)

trials = [
    ('BL_Opt_qwen2_5', 'qwen2.5:32b-instruct-q4_0-8k', 24, ['baseline_opt_1.txt', 'baseline_opt_2.txt', 'baseline_opt_3.txt', 'baseline_opt_4.txt'], run_baseline_trial),
    ('BL_Opt_gemma2', 'gemma2:27b-instruct-q4_0-8k', 24, ['baseline_opt_1.txt', 'baseline_opt_2.txt', 'baseline_opt_3.txt', 'baseline_opt_4.txt'], run_baseline_trial),
    ('BL_Opt_phi3', 'phi3:14b-medium-128k-instruct-fp16-8k', 24, ['baseline_opt_1.txt', 'baseline_opt_2.txt', 'baseline_opt_3.txt', 'baseline_opt_4.txt'], run_baseline_trial),
    ('25_BL_Opt_qwen2_5', 'qwen2.5:32b-instruct-q4_0-8k', 25, ['baseline_opt_1.txt', 'baseline_opt_2.txt', 'baseline_opt_3.txt', 'baseline_opt_4.txt'], run_baseline_trial),
    ('25_BL_Opt_gemma2', 'gemma2:27b-instruct-q4_0-8k', 25, ['baseline_opt_1.txt', 'baseline_opt_2.txt', 'baseline_opt_3.txt', 'baseline_opt_4.txt'], run_baseline_trial),
    ('25_BL_Opt_phi3', 'phi3:14b-medium-128k-instruct-fp16-8k', 25, ['baseline_opt_1.txt', 'baseline_opt_2.txt', 'baseline_opt_3.txt', 'baseline_opt_4.txt'], run_baseline_trial),
    ('26_BL_Opt_qwen2_5', 'qwen2.5:32b-instruct-q4_0-8k', 26, ['baseline_opt_1.txt', 'baseline_opt_2.txt', 'baseline_opt_3.txt', 'baseline_opt_4.txt'], run_baseline_trial),
    ('26_BL_Opt_gemma2', 'gemma2:27b-instruct-q4_0-8k', 26, ['baseline_opt_1.txt', 'baseline_opt_2.txt', 'baseline_opt_3.txt', 'baseline_opt_4.txt'], run_baseline_trial),
    ('26_BL_Opt_phi3', 'phi3:14b-medium-128k-instruct-fp16-8k', 26, ['baseline_opt_1.txt', 'baseline_opt_2.txt', 'baseline_opt_3.txt', 'baseline_opt_4.txt'], run_baseline_trial),
    ('27_BL_Opt_qwen2_5', 'qwen2.5:32b-instruct-q4_0-8k', 27, ['baseline_opt_1.txt', 'baseline_opt_2.txt', 'baseline_opt_3.txt', 'baseline_opt_4.txt'], run_baseline_trial),
    ('27_BL_Opt_gemma2', 'gemma2:27b-instruct-q4_0-8k', 27, ['baseline_opt_1.txt', 'baseline_opt_2.txt', 'baseline_opt_3.txt', 'baseline_opt_4.txt'], run_baseline_trial),
    ('27_BL_Opt_phi3', 'phi3:14b-medium-128k-instruct-fp16-8k', 27, ['baseline_opt_1.txt', 'baseline_opt_2.txt', 'baseline_opt_3.txt', 'baseline_opt_4.txt'], run_baseline_trial),
    ('28_BL_Opt_qwen2_5', 'qwen2.5:32b-instruct-q4_0-8k', 28, ['baseline_opt_1.txt', 'baseline_opt_2.txt', 'baseline_opt_3.txt', 'baseline_opt_4.txt'], run_baseline_trial),
    ('28_BL_Opt_gemma2', 'gemma2:27b-instruct-q4_0-8k', 28, ['baseline_opt_1.txt', 'baseline_opt_2.txt', 'baseline_opt_3.txt', 'baseline_opt_4.txt'], run_baseline_trial),
    ('28_BL_Opt_phi3', 'phi3:14b-medium-128k-instruct-fp16-8k', 28, ['baseline_opt_1.txt', 'baseline_opt_2.txt', 'baseline_opt_3.txt', 'baseline_opt_4.txt'], run_baseline_trial),
    ('BL_Raw_qwen2_5', 'qwen2.5:32b-instruct-q4_0-8k', 24, ['baseline_raw_1.txt', 'baseline_raw_2.txt', 'baseline_raw_3.txt', 'baseline_raw_4.txt'], run_baseline_trial),
    ('BL_Raw_gemma2', 'gemma2:27b-instruct-q4_0-8k', 24, ['baseline_raw_1.txt', 'baseline_raw_2.txt', 'baseline_raw_3.txt', 'baseline_raw_4.txt'], run_baseline_trial),
    ('BL_Raw_phi3', 'phi3:14b-medium-128k-instruct-fp16-8k', 24, ['baseline_raw_1.txt', 'baseline_raw_2.txt', 'baseline_raw_3.txt', 'baseline_raw_4.txt'], run_baseline_trial),
    ('25_BL_Raw_qwen2_5', 'qwen2.5:32b-instruct-q4_0-8k', 25, ['baseline_raw_1.txt', 'baseline_raw_2.txt', 'baseline_raw_3.txt', 'baseline_raw_4.txt'], run_baseline_trial),
    ('25_BL_Raw_gemma2', 'gemma2:27b-instruct-q4_0-8k', 25, ['baseline_raw_1.txt', 'baseline_raw_2.txt', 'baseline_raw_3.txt', 'baseline_raw_4.txt'], run_baseline_trial),
    ('25_BL_Raw_phi3', 'phi3:14b-medium-128k-instruct-fp16-8k', 25, ['baseline_raw_1.txt', 'baseline_raw_2.txt', 'baseline_raw_3.txt', 'baseline_raw_4.txt'], run_baseline_trial),
    ('26_BL_Raw_qwen2_5', 'qwen2.5:32b-instruct-q4_0-8k', 26, ['baseline_raw_1.txt', 'baseline_raw_2.txt', 'baseline_raw_3.txt', 'baseline_raw_4.txt'], run_baseline_trial),
    ('26_BL_Raw_gemma2', 'gemma2:27b-instruct-q4_0-8k', 26, ['baseline_raw_1.txt', 'baseline_raw_2.txt', 'baseline_raw_3.txt', 'baseline_raw_4.txt'], run_baseline_trial),
    ('26_BL_Raw_phi3', 'phi3:14b-medium-128k-instruct-fp16-8k', 26, ['baseline_raw_1.txt', 'baseline_raw_2.txt', 'baseline_raw_3.txt', 'baseline_raw_4.txt'], run_baseline_trial),
    ('27_BL_Raw_qwen2_5', 'qwen2.5:32b-instruct-q4_0-8k', 27, ['baseline_raw_1.txt', 'baseline_raw_2.txt', 'baseline_raw_3.txt', 'baseline_raw_4.txt'], run_baseline_trial),
    ('27_BL_Raw_gemma2', 'gemma2:27b-instruct-q4_0-8k', 27, ['baseline_raw_1.txt', 'baseline_raw_2.txt', 'baseline_raw_3.txt', 'baseline_raw_4.txt'], run_baseline_trial),
    ('27_BL_Raw_phi3', 'phi3:14b-medium-128k-instruct-fp16-8k', 27, ['baseline_raw_1.txt', 'baseline_raw_2.txt', 'baseline_raw_3.txt', 'baseline_raw_4.txt'], run_baseline_trial),
    ('28_BL_Raw_qwen2_5', 'qwen2.5:32b-instruct-q4_0-8k', 28, ['baseline_raw_1.txt', 'baseline_raw_2.txt', 'baseline_raw_3.txt', 'baseline_raw_4.txt'], run_baseline_trial),
    ('28_BL_Raw_gemma2', 'gemma2:27b-instruct-q4_0-8k', 28, ['baseline_raw_1.txt', 'baseline_raw_2.txt', 'baseline_raw_3.txt', 'baseline_raw_4.txt'], run_baseline_trial),
    ('28_BL_Raw_phi3', 'phi3:14b-medium-128k-instruct-fp16-8k', 28, ['baseline_raw_1.txt', 'baseline_raw_2.txt', 'baseline_raw_3.txt', 'baseline_raw_4.txt'], run_baseline_trial),
    ('DV_Opt_qwen2_5', 'qwen2.5:32b-instruct-q4_0', 24, ['diversify_opt.txt', 'recall_opt.txt', 'decode_opt.txt'], run_diversity_trial),
    ('DV_Opt_gemma2', 'gemma2:27b-instruct-q4_0', 24, ['diversify_opt.txt', 'recall_opt.txt', 'decode_opt.txt'], run_diversity_trial),
    ('DV_Opt_phi3', 'phi3:14b-medium-128k-instruct-fp16', 24, ['diversify_opt.txt', 'recall_opt.txt', 'decode_opt.txt'], run_diversity_trial),
    ('25_DV_Opt_qwen2_5', 'qwen2.5:32b-instruct-q4_0', 25, ['diversify_opt.txt', 'recall_opt.txt', 'decode_opt.txt'], run_diversity_trial),
    ('25_DV_Opt_gemma2', 'gemma2:27b-instruct-q4_0', 25, ['diversify_opt.txt', 'recall_opt.txt', 'decode_opt.txt'], run_diversity_trial),
    ('25_DV_Opt_phi3', 'phi3:14b-medium-128k-instruct-fp16', 25, ['diversify_opt.txt', 'recall_opt.txt', 'decode_opt.txt'], run_diversity_trial),
    ('26_DV_Opt_qwen2_5', 'qwen2.5:32b-instruct-q4_0', 26, ['diversify_opt.txt', 'recall_opt.txt', 'decode_opt.txt'], run_diversity_trial),
    ('26_DV_Opt_gemma2', 'gemma2:27b-instruct-q4_0', 26, ['diversify_opt.txt', 'recall_opt.txt', 'decode_opt.txt'], run_diversity_trial),
    ('26_DV_Opt_phi3', 'phi3:14b-medium-128k-instruct-fp16', 26, ['diversify_opt.txt', 'recall_opt.txt', 'decode_opt.txt'], run_diversity_trial),
    ('27_DV_Opt_qwen2_5', 'qwen2.5:32b-instruct-q4_0', 27, ['diversify_opt.txt', 'recall_opt.txt', 'decode_opt.txt'], run_diversity_trial),
    ('27_DV_Opt_gemma2', 'gemma2:27b-instruct-q4_0', 27, ['diversify_opt.txt', 'recall_opt.txt', 'decode_opt.txt'], run_diversity_trial),
    ('27_DV_Opt_phi3', 'phi3:14b-medium-128k-instruct-fp16', 27, ['diversify_opt.txt', 'recall_opt.txt', 'decode_opt.txt'], run_diversity_trial),
    ('28_DV_Opt_qwen2_5', 'qwen2.5:32b-instruct-q4_0', 28, ['diversify_opt.txt', 'recall_opt.txt', 'decode_opt.txt'], run_diversity_trial),
    ('28_DV_Opt_gemma2', 'gemma2:27b-instruct-q4_0', 28, ['diversify_opt.txt', 'recall_opt.txt', 'decode_opt.txt'], run_diversity_trial),
    ('28_DV_Opt_phi3', 'phi3:14b-medium-128k-instruct-fp16', 28, ['diversify_opt.txt', 'recall_opt.txt', 'decode_opt.txt'], run_diversity_trial),
    ('DV_Raw_qwen2_5', 'qwen2.5:32b-instruct-q4_0', 24, ['diversify_raw.txt', 'recall_raw.txt', 'decode_raw.txt'], run_diversity_trial),
    ('DV_Raw_gemma2', 'gemma2:27b-instruct-q4_0', 24, ['diversify_raw.txt', 'recall_raw.txt', 'decode_raw.txt'], run_diversity_trial),
    ('DV_Raw_phi3', 'phi3:14b-medium-128k-instruct-fp16', 24, ['diversify_raw.txt', 'recall_raw.txt', 'decode_raw.txt'], run_diversity_trial),
    ('25_DV_Raw_qwen2_5', 'qwen2.5:32b-instruct-q4_0', 25, ['diversify_raw.txt', 'recall_raw.txt', 'decode_raw.txt'], run_diversity_trial),
    ('25_DV_Raw_gemma2', 'gemma2:27b-instruct-q4_0', 25, ['diversify_raw.txt', 'recall_raw.txt', 'decode_raw.txt'], run_diversity_trial),
    ('25_DV_Raw_phi3', 'phi3:14b-medium-128k-instruct-fp16', 25, ['diversify_raw.txt', 'recall_raw.txt', 'decode_raw.txt'], run_diversity_trial),
    ('26_DV_Raw_qwen2_5', 'qwen2.5:32b-instruct-q4_0', 26, ['diversify_raw.txt', 'recall_raw.txt', 'decode_raw.txt'], run_diversity_trial),
    ('26_DV_Raw_gemma2', 'gemma2:27b-instruct-q4_0', 26, ['diversify_raw.txt', 'recall_raw.txt', 'decode_raw.txt'], run_diversity_trial),
    ('26_DV_Raw_phi3', 'phi3:14b-medium-128k-instruct-fp16', 26, ['diversify_raw.txt', 'recall_raw.txt', 'decode_raw.txt'], run_diversity_trial),
    ('27_DV_Raw_qwen2_5', 'qwen2.5:32b-instruct-q4_0', 27, ['diversify_raw.txt', 'recall_raw.txt', 'decode_raw.txt'], run_diversity_trial),
    ('27_DV_Raw_gemma2', 'gemma2:27b-instruct-q4_0', 27, ['diversify_raw.txt', 'recall_raw.txt', 'decode_raw.txt'], run_diversity_trial),
    ('27_DV_Raw_phi3', 'phi3:14b-medium-128k-instruct-fp16', 27, ['diversify_raw.txt', 'recall_raw.txt', 'decode_raw.txt'], run_diversity_trial),
    ('28_DV_Raw_qwen2_5', 'qwen2.5:32b-instruct-q4_0', 28, ['diversify_raw.txt', 'recall_raw.txt', 'decode_raw.txt'], run_diversity_trial),
    ('28_DV_Raw_gemma2', 'gemma2:27b-instruct-q4_0', 28, ['diversify_raw.txt', 'recall_raw.txt', 'decode_raw.txt'], run_diversity_trial),
    ('28_DV_Raw_phi3', 'phi3:14b-medium-128k-instruct-fp16', 28, ['diversify_raw.txt', 'recall_raw.txt', 'decode_raw.txt'], run_diversity_trial)
]

if __name__ == '__main__':
    for trial in trials:
        print(trial)
        _inference_cache = {}
        trial[4](trial[0], trial[1], trial[2], trial[3])

