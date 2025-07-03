import json
import statistics
import os

from collections import defaultdict

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

MUTATION_NAMES = [
    'Crossover',
    'Guidelines',
    'Expert',
    'Structure',
    'Constraints',
    'Backstory',
    'SplitInstructions',
    'Streamline',
    'DefineRole',
    'Positivity',
    'AddExample',
    'PromptEngineer',
    'GuidedEvolution'
]

def hamming(a, b, threshold=0.001):
    if len(a) != len(b):
        raise ValueError("Lists must be of the same length.")
    mismatches = sum(abs(x - y) > threshold for x, y in zip(a, b))
    return mismatches / len(a)

def group_scores_by_suggestion(pairs, grouped_scores, grouped_diversities, prompt_scores, criteria_scores):
    for score, suggestion, uuid, p, c in pairs:
        matched = False
        parent_score = sum([prompt_scores[uuid] for uuid in p]) / len(p)
        delta_score = score - parent_score
        diversity = sum(hamming(c, criteria_scores[uuid]) for uuid in p) / len(p)

        for idx, mutation in enumerate(MUTATIONS):
            if suggestion.strip() == mutation:
                grouped_scores[mutation].append(delta_score)
                grouped_diversities[mutation].append(diversity)
                matched = True
                break

        if not matched:
            if suggestion.startswith("Crossover"):
                grouped_scores["Crossover"].append(delta_score)
                grouped_diversities["Crossover"].append(diversity)
            else:
                grouped_scores["GuidedEvolution"].append(delta_score)
                grouped_diversities["GuidedEvolution"].append(diversity)

    return grouped_scores

def extract_score_suggestion_pairs(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)

    pairs = []
    for uuid, entry in data.items():
        score = entry.get("score")
        suggestion = entry.get("suggestion")
        parents = entry.get("parents")
        c = entry.get("criteria_scores")
        if score is not None:
            pairs.append((score, suggestion, uuid, parents, c))

    return pairs

def success_rate(data):
    return len([x for x in data if x > 0]) / len(data)

def calculate_treatment_scores(basedirs):
    treatment_scores = defaultdict(list)
    treatment_diversities = defaultdict(list)

    for basedir in basedirs:
        baseline = extract_score_suggestion_pairs(f"{basedir}/0/meta.json")
        prompt_scores = {uuid: score for score, _, uuid, _, _ in baseline}
        criteria_scores = {uuid: c for _, _, uuid, _, c in baseline}
        generations = sorted([int(d) for d in os.listdir(basedir) if d.isdigit()])

        for i in range(1, max(generations) + 1):
            treatment = extract_score_suggestion_pairs(f"{basedir}/{i}/meta.json")
            prompt_scores.update({uuid: score for score, _, uuid, _, _ in treatment})
            criteria_scores.update({uuid: c for _, _, uuid, _, c in treatment})

            group_scores_by_suggestion(treatment, treatment_scores, treatment_diversities, prompt_scores,
                                       criteria_scores)

    return treatment_scores, treatment_diversities

if __name__ == "__main__":
    all_scores, all_diversities = calculate_treatment_scores(['diversify', 'recall', 'decode', 'baseline'])
    div_scores, _ = calculate_treatment_scores(['diversify'])
    rec_scores, _ = calculate_treatment_scores(['recall'])
    dec_scores, _ = calculate_treatment_scores(['decode'])
    bas_scores, _ = calculate_treatment_scores(['baseline'])

    def format_line(operator, key):
        s = f"{operator:<20}"
        for scores in [div_scores, rec_scores, dec_scores, bas_scores]:
            s += f"\t{statistics.median(scores[key]):10.2f}\t{max(scores[key]):10.2f}"
        s += f"\t{success_rate(all_scores[key]):10.2f}\t{statistics.mean(all_diversities[key]):10.2f}"
        print(s)

    print(f"{'':<20}\t{'Diversify':^20}\t{'Recall':^20}\t{'Decode':^20}\t{'Baseline':^20}\t{'Aggregate':^20}")
    print(f"{'Operator':<20}\t{'Median':>10}\t{'Max':>10}\t{'Median':>10}\t{'Max':>10}\t{'Median':>10}\t{'Max':>10}\t{'Median':>10}\t{'Max':>10}\t{'Success':>10}\t{'Exploration':>10}")

    format_line('Crossover', 'Crossover')
    for idx, mutation in enumerate(MUTATIONS):
        format_line(MUTATION_NAMES[idx + 1], mutation)
    format_line('GuidedEvolution', 'GuidedEvolution')