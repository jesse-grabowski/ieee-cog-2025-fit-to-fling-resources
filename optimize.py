import os
import json
import random
import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
from uuid import uuid4
from collections import defaultdict
from openai import OpenAI
from string import Template
from dotenv import load_dotenv
from tqdm import tqdm
from itertools import combinations
from sklearn.metrics.pairwise import cosine_distances
import numpy as np

load_dotenv()

# optimize here, expects PROMPTS_DIR/0/meta.json to exist
PROMPTS_DIR = "prompts"
# how many mutations to apply to each parent per generation
NUM_MUTATIONS = 4
# how many parents to select
NUM_CANDIDATES = 4
# stop evaluating early if a prompt won't hit CULL_THRESHOLD * <previous generation P_p lowest score>
# 0.01 is effectively off, 0.7-0.9 or so is reasonable but doesn't make pretty data for plots and tables
CULL_THRESHOLD = 0.01

# Cheap but terrible: gpt-4.1-nano
# Good cost performance: gpt-4.1-mini
# Reasonable overall but worse than gpt-4.1-mini: gpt-4o-mini
# teacher model
OPENAI_MODEL = 'gpt-4.1-mini'
cloud_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.openai.com/v1"
)

# evaluation model
OLLAMA_MODEL = 'phi3:14b-medium-128k-instruct-fp16-8k'
ollama_client = OpenAI(
    api_key="ollama",  # dummy key
    base_url="http://localhost:11434/v1"
)

class EvaluationStrictness(Enum):
  EVAL_STRICT = 0
  EVAL_LAST_CODE_BLOCK = 1
  EVAL_STRICT_REWARD_PARTIAL = 2

evaluation_strictness = EvaluationStrictness.EVAL_LAST_CODE_BLOCK
evaluation_input = """Input: ${content}
"""
evaluation_timeout = 90
evaluation_criteria = [
  {
    "vars": {
      "content": "A"
    },
    "expectation": "```\ndrop_block('b11', 8)\ndrop_block('b11', 10)\ndrop_block('b11', 8)\ndrop_block('b11', 10)\ndrop_block('b31', 9)\ndrop_block('b11', 8)\ndrop_block('b11', 10)\ndrop_block('b31', 9)\n```"
  },
  {
    "vars": {
      "content": "B"
    },
    "expectation": "```\ndrop_block('b31', 9)\ndrop_block('b11', 11)\ndrop_block('b13', 8)\ndrop_block('b13', 10)\ndrop_block('b13', 11)\ndrop_block('b31', 9)\ndrop_block('b11', 11)\ndrop_block('b11', 8)\ndrop_block('b11', 10)\ndrop_block('b31', 9)\n```"
  },
  {
    "vars": {
      "content": "C"
    },
    "expectation": "```\ndrop_block('b31', 10)\ndrop_block('b13', 9)\ndrop_block('b13', 10)\ndrop_block('b13', 9)\ndrop_block('b13', 10)\ndrop_block('b31', 10)\n```"
  },
  {
    "vars": {
      "content": "D"
    },
    "expectation": "```\ndrop_block('b31', 10)\ndrop_block('b13', 12)\ndrop_block('b13', 12)\ndrop_block('b13', 9)\ndrop_block('b13', 10)\ndrop_block('b13', 9)\ndrop_block('b13', 10)\ndrop_block('b11', 12)\ndrop_block('b31', 10)\n```"
  },
  {
    "vars": {
      "content": "E"
    },
    "expectation": "```\ndrop_block('b31', 10)\ndrop_block('b11', 9)\ndrop_block('b11', 10)\ndrop_block('b31', 10)\ndrop_block('b11', 9)\ndrop_block('b11', 10)\ndrop_block('b31', 10)\n```"
  },
  {
    "vars": {
      "content": "F"
    },
    "expectation": "```\ndrop_block('b13', 10)\ndrop_block('b13', 11)\ndrop_block('b13', 10)\ndrop_block('b13', 11)\ndrop_block('b31', 11)\ndrop_block('b11', 10)\ndrop_block('b11', 11)\ndrop_block('b31', 11)\n```"
  },
  {
    "vars": {
      "content": "G"
    },
    "expectation": "```\ndrop_block('b13', 6)\ndrop_block('b13', 7)\ndrop_block('b31', 9)\ndrop_block('b31', 9)\ndrop_block('b31', 12)\ndrop_block('b31', 12)\ndrop_block('b11', 12)\ndrop_block('b11', 13)\ndrop_block('b31', 12)\ndrop_block('b13', 6)\ndrop_block('b13', 7)\ndrop_block('b11', 6)\ndrop_block('b11', 7)\ndrop_block('b31', 7)\n```"
  },
  {
    "vars": {
      "content": "H"
    },
    "expectation": "```\ndrop_block('b13', 9)\ndrop_block('b13', 11)\ndrop_block('b31', 10)\ndrop_block('b13', 9)\ndrop_block('b13', 11)\n```"
  },
  {
    "vars": {
      "content": "I"
    },
    "expectation": "```\ndrop_block('b13', 10)\ndrop_block('b13', 10)\n```"
  },
  {
    "vars": {
      "content": "J"
    },
    "expectation": "```\ndrop_block('b11', 10)\ndrop_block('b13', 11)\ndrop_block('b13', 11)\n```"
  },
  {
    "vars": {
      "content": "K"
    },
    "expectation": "```\ndrop_block('b13', 9)\ndrop_block('b13', 11)\ndrop_block('b11', 12)\ndrop_block('b31', 10)\ndrop_block('b31', 11)\ndrop_block('b13', 9)\ndrop_block('b11', 12)\ndrop_block('b11', 12)\n```"
  },
  {
    "vars": {
      "content": "L"
    },
    "expectation": "```\ndrop_block('b13', 10)\ndrop_block('b13', 10)\ndrop_block('b11', 11)\n```"
  },
  {
    "vars": {
      "content": "M"
    },
    "expectation": "```\ndrop_block('b13', 5)\ndrop_block('b13', 6)\ndrop_block('b13', 8)\ndrop_block('b13', 10)\ndrop_block('b13', 11)\ndrop_block('b13', 13)\ndrop_block('b13', 14)\ndrop_block('b13', 5)\ndrop_block('b13', 6)\ndrop_block('b13', 8)\ndrop_block('b13', 10)\ndrop_block('b13', 11)\ndrop_block('b13', 13)\ndrop_block('b13', 14)\ndrop_block('b31', 6)\ndrop_block('b11', 8)\ndrop_block('b31', 10)\ndrop_block('b31', 13)\n```"
  },
  {
    "vars": {
      "content": "N"
    },
    "expectation": "```\ndrop_block('b13', 9)\ndrop_block('b13', 11)\ndrop_block('b13', 9)\ndrop_block('b13', 11)\ndrop_block('b31', 10)\ndrop_block('b11', 9)\ndrop_block('b11', 11)\n```"
  },
  {
    "vars": {
      "content": "O"
    },
    "expectation": "```\ndrop_block('b31', 10)\ndrop_block('b11', 9)\ndrop_block('b11', 11)\ndrop_block('b11', 9)\ndrop_block('b11', 11)\ndrop_block('b11', 9)\ndrop_block('b11', 11)\ndrop_block('b31', 10)\n```"
  },
  {
    "vars": {
      "content": "P"
    },
    "expectation": "```\ndrop_block('b13', 9)\ndrop_block('b13', 10)\ndrop_block('b13', 9)\ndrop_block('b13', 10)\ndrop_block('b31', 10)\ndrop_block('b13', 9)\ndrop_block('b13', 11)\ndrop_block('b31', 10)\n```"
  },
  {
    "vars": {
      "content": "Q"
    },
    "expectation": "```\ndrop_block('b31', 10)\ndrop_block('b13', 9)\ndrop_block('b13', 9)\ndrop_block('b31', 11)\ndrop_block('b11', 11)\ndrop_block('b13', 11)\ndrop_block('b11', 11)\ndrop_block('b31', 10)\n```"
  },
  {
    "vars": {
      "content": "R"
    },
    "expectation": "```\ndrop_block('b13', 7)\ndrop_block('b13', 8)\ndrop_block('b13', 10)\ndrop_block('b31', 8)\ndrop_block('b13', 7)\ndrop_block('b13', 9)\ndrop_block('b11', 10)\ndrop_block('b31', 8)\n```"
  },
  {
    "vars": {
      "content": "S"
    },
    "expectation": "```\ndrop_block('b31', 9)\ndrop_block('b11', 9)\ndrop_block('b11', 10)\ndrop_block('b31', 9)\ndrop_block('b11', 8)\ndrop_block('b11', 9)\ndrop_block('b31', 9)\n```"
  },
  {
    "vars": {
      "content": "T"
    },
    "expectation": "```\ndrop_block('b13', 10)\ndrop_block('b13', 10)\ndrop_block('b31', 10)\n```"
  },
  {
    "vars": {
      "content": "U"
    },
    "expectation": "```\ndrop_block('b31', 11)\ndrop_block('b31', 8)\ndrop_block('b13', 7)\ndrop_block('b13', 12)\n```"
  },
  {
    "vars": {
      "content": "V"
    },
    "expectation": "```\ndrop_block('b13', 9)\ndrop_block('b13', 8)\ndrop_block('b13', 10)\ndrop_block('b13', 7)\ndrop_block('b13', 11)\ndrop_block('b31', 7)\ndrop_block('b31', 11)\ndrop_block('b11', 9)\ndrop_block('b31', 9)\ndrop_block('b31', 9)\ndrop_block('b13', 6)\ndrop_block('b13', 7)\ndrop_block('b13', 11)\ndrop_block('b13', 12)\ndrop_block('b13', 6)\ndrop_block('b13', 7)\ndrop_block('b13', 11)\n```"
  },
  {
    "vars": {
      "content": "W"
    },
    "expectation": "```\ndrop_block('b11', 9)\ndrop_block('b11', 10)\ndrop_block('b11', 11)\ndrop_block('b11', 12)\ndrop_block('b31', 9)\ndrop_block('b31', 12)\ndrop_block('b11', 10)\ndrop_block('b11', 11)\ndrop_block('b11', 9)\ndrop_block('b11', 12)\ndrop_block('b11', 9)\ndrop_block('b11', 12)\ndrop_block('b11', 9)\ndrop_block('b11', 12)\ndrop_block('b11', 9)\ndrop_block('b11', 12)\ndrop_block('b11', 9)\ndrop_block('b11', 12)\n```"
  },
  {
    "vars": {
      "content": "X"
    },
    "expectation": "```\ndrop_block('b13', 9)\ndrop_block('b13', 11)\ndrop_block('b31', 10)\ndrop_block('b11', 10)\ndrop_block('b31', 10)\ndrop_block('b13', 9)\ndrop_block('b13', 11)\n```"
  },
  {
    "vars": {
      "content": "Y"
    },
    "expectation": "```\ndrop_block('b13', 9)\ndrop_block('b31', 9)\ndrop_block('b11', 8)\ndrop_block('b11', 10)\ndrop_block('b11', 8)\ndrop_block('b11', 10)\ndrop_block('b11', 8)\ndrop_block('b11', 10)\n```"
  },
  {
    "vars": {
      "content": "Z"
    },
    "expectation": "```\ndrop_block('b31', 8)\ndrop_block('b31', 8)\ndrop_block('b31', 11)\ndrop_block('b31', 11)\ndrop_block('b11', 7)\ndrop_block('b11', 8)\ndrop_block('b31', 8)\ndrop_block('b31', 8)\ndrop_block('b11', 8)\ndrop_block('b11', 9)\ndrop_block('b31', 8)\ndrop_block('b31', 8)\n```"
  }
]

MUTATIONS = [
"Add detailed guidelines or modify existing instructions to improve specificity",
"Introduce an expert persona or change the existing persona to further emphasize the agent's expertise",
"Modify the prompt's structure or architecture. This could involve splitting or merging sections, changing markdown elements used, or other structural changes",
"Introduce new constraints or rephrase existing ones",
"Introduce a creative backstory for the agent",
"Break complex instructions down into smaller steps",
"Streamline the prompt by condensing instructions and eliminating redundancy, while preserving essential elements like lookup tables, instructions, and the task itself.",
"Assign the agent a well-defined role or behavior",
"Rephrase the prompt, replacing negative statements like \"do not do X\" with positive statements like \"only do Y\"",
"Add a new example or modify an existing example to cover the given errors",
"Incorporate established prompting techniques such as chain of thought or reason + act (ReAct) to enhance clarity and decision-making"
]

GRADIENT_TEMPLATE = """You will be given some errors (`==========ERRORS==========`), followed by a prompt (`==========PROMPT==========`). Offer two high-impact suggestions on how to improve the prompt to avoid the given errors. Between each suggestion, output `==========`. Each suggestion should be a single paragraph. Do not number your suggestions. Output only your suggestions and no other text.

==========ERRORS==========

${errors}

==========PROMPT==========

${prompt}"""

SUGGESTION_TEMPLATE = """You will be given some errors (`==========ERRORS==========`), followed by a prompt (`==========PROMPT==========`). Improve the prompt for evaluation on Phi3 to mitigate the given errors by implementing the following suggestion:

Suggestion: ${suggestion}

Output exactly the improved prompt and nothing else. Display as markdown, but do not wrap the output in code fences:

==========ERRORS==========

${errors}

==========PROMPT==========

${prompt}"""

ERROR_TEMPLATE = """==========INPUT==========

${input}

==========EXPECTED==========

${expected}

==========ACTUAL==========

${actual}"""

CROSSOVER_TEMPLATE = """You will be given two prompts (`==========PROMPT1==========` and `==========PROMPT2==========`). Combine the two prompts, taking elements from each to create a new, improved prompt. This might involve taking the role from one prompt and the instructions from the other, mixing examples, mixing instructions, or other combinations. Output exactly the improved prompt and nothing else. Display as markdown, but do not wrap the output in code fences:

==========PROMPT1==========

${prompt1}

==========PROMPT2==========

${prompt2}"""

@dataclass
class Prompt:
    text: str
    generation: int
    uuid: str
    score: float
    criteria_scores: List[int]
    responses: List[str]
    parents: List[str]
    suggestion: Optional[str] = None


def get_generation_dirs(base_dir: str) -> List[int]:
    return sorted(
        [int(d) for d in os.listdir(base_dir) if d.isdigit() and os.path.isdir(os.path.join(base_dir, d))]
    )


def load_prompts_from_generation(generation: int) -> List[Prompt]:
    gen_dir = os.path.join(PROMPTS_DIR, str(generation))
    meta_path = os.path.join(gen_dir, "meta.json")

    if not os.path.exists(meta_path):
        print(f"Warning: meta.json not found in {gen_dir}")
        return []

    with open(meta_path, "r", encoding="utf-8") as meta_file:
        meta = json.load(meta_file)

    prompts = []
    for uuid_str, info in meta.items():
        prompt_file = os.path.join(gen_dir, f"{uuid_str}.txt")
        if not os.path.exists(prompt_file):
            print(f"Warning: Missing prompt file {prompt_file}")
            continue
        with open(prompt_file, "r", encoding="utf-8") as f:
            text = f.read()
        prompts.append(Prompt(text=text, generation=generation, uuid=uuid_str, score=info.get("score", -1), parents=info.get("parents", []), suggestion=info.get("suggestion", None), criteria_scores=info.get("criteria_scores", [0] * len(evaluation_criteria)), responses=info.get("responses", [''] * len(evaluation_criteria))))

    return prompts


def load_all_prompts() -> List[Prompt]:
    all_prompts = []
    for generation in get_generation_dirs(PROMPTS_DIR):
        all_prompts.extend(load_prompts_from_generation(generation))
    return all_prompts


def save_prompts(prompts: List[Prompt]):
    if not prompts:
        print("No prompts to save.")
        return

    grouped_prompts = defaultdict(list)
    for prompt in prompts:
        grouped_prompts[prompt.generation].append(prompt)

    for generation, gen_prompts in grouped_prompts.items():
        gen_dir = os.path.join(PROMPTS_DIR, str(generation))
        os.makedirs(gen_dir, exist_ok=True)

        meta = {}
        for prompt in gen_prompts:
            prompt_file = os.path.join(gen_dir, f"{prompt.uuid}.txt")
            with open(prompt_file, "w", encoding="utf-8") as f:
                f.write(prompt.text)
            meta[prompt.uuid] = {
                "score": prompt.score,
                "parents": prompt.parents,
                "suggestion": prompt.suggestion or None,
                "criteria_scores": prompt.criteria_scores or ([0] * len(evaluation_criteria)),
                "responses": prompt.responses or ([''] * len(evaluation_criteria)),
            }

        meta_file = os.path.join(gen_dir, "meta.json")
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

def get_latest_generation(prompts: List[Prompt]) -> int:
    if not prompts:
        raise ValueError("Prompt list is empty.")
    return max(p.generation for p in prompts)

def top_n_prompts(prompts: List[Prompt], n: int = 8) -> List[Prompt]:
    latest_generation = get_latest_generation(prompts)
    filtered_prompts = [p for p in prompts if p.score > 0]
    random.shuffle(filtered_prompts)
    best_prompts = sorted(
        filtered_prompts,
        key=lambda p: (p.score * .95 ** (latest_generation - p.generation), p.generation),
        reverse=True
    )[:n * 3]
    if len(best_prompts) == 0:
      return prompts[:n]
    elif len(best_prompts) <= n:
      return best_prompts
    else:
      return pick_diverse_prompts(best_prompts, n)

def pick_diverse_prompts(prompts: List[Prompt], n: int) -> List[Prompt]:
  max_index = max(range(len(prompts)), key=lambda i: prompts[i].score)
  remaining_indices = [i for i in range(len(prompts)) if i != max_index]

  vectors = np.array([p.criteria_scores for p in prompts])
  dist_matrix = cosine_distances(vectors)

  max_solution_score = -1
  max_total_distance = -1
  best_combination = None

  for combo in combinations(remaining_indices, n - 1):
    indices = (max_index,) + combo
    total_distance = sum(
      dist_matrix[i][j]
      for i, j in combinations(indices, 2)
    )
    selected_prompts = [prompts[i] for i in indices]
    transposed_scores = zip(*(p.criteria_scores for p in selected_prompts))
    solution_score = sum(max(scores_at_index) for scores_at_index in transposed_scores)
    if solution_score > max_solution_score or (solution_score == max_solution_score and total_distance > max_total_distance):
      max_solution_score = solution_score
      max_total_distance = total_distance
      best_combination = selected_prompts

  return best_combination

def evolve(candidates: List[Prompt], generation: int) -> List[Prompt]:
    next_generation : List[Prompt] = []
    next_generation.extend(crossover(candidates, generation))
    for prompt in candidates:
        next_generation.extend(expand(prompt, generation))
        next_generation.extend(mutate(prompt, generation, n = NUM_MUTATIONS))
    return next_generation

def format_errors(prompt: Prompt, n: int = 2) -> str:
    input_template: Template = Template(evaluation_input)
    error_template: Template = Template(ERROR_TEMPLATE)

    failed_indices = [(prompt.criteria_scores[i], random.random(), i) for i in range(len(prompt.criteria_scores))]
    failed_indices = list(filter(lambda indice: indice[0] < 1, failed_indices))
    failed_indices.sort()
    failed_criteria = [(evaluation_criteria[i[2]], prompt.responses[i[2]]) for i in failed_indices[:n]]

    errors = [error_template.substitute(
      input=input_template.substitute(criteria['vars']),
      expected=criteria['expectation'],
      actual=response
    ) for criteria, response in failed_criteria]

    return '\n\n'.join(errors) or ''

def mutate(prompt: Prompt, generation: int, n = 8) -> List[Prompt]:
    results : List[Prompt] = []
    mutations = random.sample(MUTATIONS, n)

    for mutation in mutations:
        template: Template = Template(SUGGESTION_TEMPLATE)
        chat_completion = cloud_client.chat.completions.create(
            messages=[{"role": "user", "content": template.substitute(prompt=prompt.text, errors=format_errors(prompt), suggestion=mutation)}],
            model=OPENAI_MODEL,
            n=1,
        )
        results.append(Prompt(chat_completion.choices[0].message.content, generation, str(uuid4()), -1,
                              parents=[prompt.uuid],
                              suggestion=mutation,
                              criteria_scores=([0] * len(evaluation_criteria)),
                              responses=([''] * len(evaluation_criteria))))

    return results

def expand(prompt: Prompt, generation: int) -> List[Prompt]:
    gradient_template = Template(GRADIENT_TEMPLATE)

    chat_completion = cloud_client.chat.completions.create(
        messages=[{"role": "user", "content": gradient_template.substitute(errors=format_errors(prompt), prompt=prompt.text)}],
        model=OPENAI_MODEL,
        n=1,
    )
    suggestions = chat_completion.choices[0].message.content
    suggestions = suggestions.split("==========")
    # Just do something random if our teacher model includes a trailing delimiter
    suggestions = [s.strip() or "Completely rewrite the prompt" for s in suggestions]

    results : List[Prompt] = []

    for suggestion in suggestions:
        template = Template(SUGGESTION_TEMPLATE)
        chat_completion = cloud_client.chat.completions.create(
            messages=[{"role": "user", "content": template.substitute(suggestion=suggestion, errors=format_errors(prompt), prompt=prompt.text)}],
            model=OPENAI_MODEL,
            n=1,
        )
        results.append(Prompt(chat_completion.choices[0].message.content, generation, str(uuid4()), -1,
                              parents=[prompt.uuid],
                              suggestion=suggestion,
                              criteria_scores=[0] * len(evaluation_criteria),
                              responses=([''] * len(evaluation_criteria))))

    return results

def crossover(prompts: List[Prompt], generation: int) -> List[Prompt]:
  results: List[Prompt] = []
  template: Template = Template(CROSSOVER_TEMPLATE)

  for prompt1, prompt2 in combinations(prompts, 2):
    chat_completion = cloud_client.chat.completions.create(
      messages=[{"role": "user", "content": template.substitute(prompt1=prompt1.text, prompt2=prompt2.text)}],
      model=OPENAI_MODEL,
      n=1,
    )
    results.append(
      Prompt(chat_completion.choices[0].message.content, generation, str(uuid4()), -1,
             parents=[prompt1.uuid, prompt2.uuid],
             suggestion=f"Crossover {prompt1.uuid} + {prompt2.uuid}",
             criteria_scores=([0] * len(evaluation_criteria)),
             responses=([''] * len(evaluation_criteria))))

  return results

def evaluate(prompt: Prompt, order = None, cull_below = None):
    if prompt.score != -1:
        return

    template: Template = Template(prompt.text)

    sorted_criteria = evaluation_criteria[:]
    original_indices = [i for i in range(0, len(evaluation_criteria))]
    if order is not None:
        sorted_criteria = [x for _, x in sorted(zip(order, sorted_criteria), key=lambda pair: pair[0])]
        original_indices = [x for _, x in sorted(zip(order, original_indices), key=lambda pair: pair[0])]

    score = 0
    criteria_scores = [0] * len(evaluation_criteria)
    # really it just took too long, but in most cases this was due to the response being too many tokens
    responses = ['<response too many tokens - inference timeout exceeded>'] * len(evaluation_criteria)
    with tqdm(zip(original_indices, sorted_criteria), desc=prompt.uuid, total=len(evaluation_criteria)) as progress:
        progress.set_postfix(score=score / len(evaluation_criteria))
        for i, (original_index, criteria) in enumerate(progress):
            if cull_below is not None and (len(evaluation_criteria) - i + score) / len(evaluation_criteria) < cull_below:
                progress.set_postfix(culled=True)
                break

            try:
                chat_completion = ollama_client.chat.completions.create(
                    messages=[{"role": "user", "content": template.substitute(criteria['vars'])}],
                    model=OLLAMA_MODEL,
                    temperature=1,
                    timeout = evaluation_timeout,
                    seed=42,
                    n=1,
                )

                response = chat_completion.choices[0].message.content.strip()

                responses[original_index] = response

                if evaluation_strictness == EvaluationStrictness.EVAL_LAST_CODE_BLOCK:
                  if extract_last_code_block(response) == criteria['expectation']:
                    score += 1
                    criteria_scores[original_index] = 1
                  elif "".join(extract_last_code_block(response).split()) == "".join(criteria['expectation'].split()):
                    score += 0.5
                    criteria_scores[original_index] = 0.5
                elif evaluation_strictness == EvaluationStrictness.EVAL_STRICT:
                  if response == criteria['expectation']:
                      score += 1
                      criteria_scores[original_index] = 1
                elif evaluation_strictness == EvaluationStrictness.EVAL_STRICT_REWARD_PARTIAL:
                  if response == criteria['expectation']:
                      score += 1
                      criteria_scores[original_index] = 1
                  elif criteria['expectation'] in response:
                      score += 0.5
                      criteria_scores[original_index] = 0.5

                progress.set_postfix(score=score / len(evaluation_criteria))
            except:
                pass

    prompt.criteria_scores = criteria_scores
    prompt.responses = responses
    prompt.score = score / len(evaluation_criteria)


def extract_last_code_block(text: str) -> str | None:
  pattern = re.compile(r"```[\w+-]*\n.*?```", re.DOTALL)
  matches = pattern.findall(text)
  return matches[-1] if matches else None

if __name__ == "__main__":
    prompts = load_all_prompts()

    candidates = top_n_prompts(prompts, n=NUM_CANDIDATES)
    cull_below = min([p.score for p in candidates], default=0) * CULL_THRESHOLD
    criteria_scores = [sum(scores) for scores in zip(*[p.criteria_scores for p in candidates])]
    for prompt in prompts:
        evaluate(prompt, order=criteria_scores or None, cull_below=cull_below if cull_below > 0 else None)
        save_prompts(prompts)

    while True:
        prompts = load_all_prompts()

        if any(p.score == 1 for p in prompts):
            exit(0)

        candidates = top_n_prompts(prompts, n = NUM_CANDIDATES)
        print('Selected parents: ', [p.uuid for p in candidates])

        generation = get_latest_generation(prompts) + 1
        next_generation = evolve(candidates, generation)
        save_prompts(next_generation)

        cull_below = min([p.score for p in candidates], default=0) * CULL_THRESHOLD
        criteria_scores = [sum(scores) for scores in zip(*[p.criteria_scores for p in candidates])]
        for prompt in next_generation:
            evaluate(prompt, order=criteria_scores, cull_below=cull_below)
            save_prompts(next_generation)