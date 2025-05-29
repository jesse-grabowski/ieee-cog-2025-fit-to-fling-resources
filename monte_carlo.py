import itertools
import json
import os
import random
import re
import string
import scienceplots
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_distances

plt.style.use(['science', 'ieee'])
random.seed(8675309)

BASEDIR = 'ablation_data'
EXPERIMENTS = [
    ('BL_Opt_gemma2', ['BL_Opt_gemma2', '25_BL_Opt_gemma2', '26_BL_Opt_gemma2', '27_BL_Opt_gemma2', '28_BL_Opt_gemma2']),
    ('BL_Opt_phi3', ['BL_Opt_phi3', '25_BL_Opt_phi3', '26_BL_Opt_phi3', '27_BL_Opt_phi3', '28_BL_Opt_phi3']),
    ('BL_Opt_qwen2_5', ['BL_Opt_qwen2_5', '25_BL_Opt_qwen2_5', '26_BL_Opt_qwen2_5', '27_BL_Opt_qwen2_5', '28_BL_Opt_qwen2_5']),
    ('BL_Raw_gemma2', ['BL_Raw_gemma2', '25_BL_Raw_gemma2', '26_BL_Raw_gemma2', '27_BL_Raw_gemma2', '28_BL_Raw_gemma2']),
    ('BL_Raw_phi3', ['BL_Raw_phi3', '25_BL_Raw_phi3', '26_BL_Raw_phi3', '27_BL_Raw_phi3', '28_BL_Raw_phi3']),
    ('BL_Raw_qwen2_5', ['BL_Raw_qwen2_5', '25_BL_Raw_qwen2_5', '26_BL_Raw_qwen2_5', '27_BL_Raw_qwen2_5', '28_BL_Raw_qwen2_5']),
    ('DV_Opt_gemma2', ['DV_Opt_gemma2', '25_DV_Opt_gemma2', '26_DV_Opt_gemma2', '27_DV_Opt_gemma2', '28_DV_Opt_gemma2']),
    ('DV_Opt_phi3', ['DV_Opt_phi3', '25_DV_Opt_phi3', '26_DV_Opt_phi3', '27_DV_Opt_phi3', '28_DV_Opt_phi3']),
    ('DV_Opt_qwen2_5', ['DV_Opt_qwen2_5', '25_DV_Opt_qwen2_5', '26_DV_Opt_qwen2_5', '27_DV_Opt_qwen2_5', '28_DV_Opt_qwen2_5']),
    ('DV_Raw_gemma2', ['DV_Raw_gemma2', '25_DV_Raw_gemma2', '26_DV_Raw_gemma2', '27_DV_Raw_gemma2', '28_DV_Raw_gemma2']),
    ('DV_Raw_phi3', ['DV_Raw_phi3', '25_DV_Raw_phi3', '26_DV_Raw_phi3', '27_DV_Raw_phi3', '28_DV_Raw_phi3']),
    ('DV_Raw_qwen2_5', ['DV_Raw_qwen2_5', '25_DV_Raw_qwen2_5', '26_DV_Raw_qwen2_5', '27_DV_Raw_qwen2_5', '28_DV_Raw_qwen2_5']),
]

def count_expected(experiment, letter):
    raw_dir = os.path.join(BASEDIR, experiment, 'raw', letter)
    return len([d for d in os.listdir(raw_dir) if os.path.isfile(os.path.join(raw_dir, d))])

def load_similarity_scores(experiment, letter, count):
    scores = [0] * count
    try:
        with open(os.path.join(BASEDIR, experiment, 'similarity', f'{letter}.json'), 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        return scores
    for trial in data['trials']:
        index = int(re.search(r'(\d+)(?=\.png$)', trial['id']).group(1)) - 1
        scores[index] = trial['similarity']
    return scores

def load_stability_scores(experiment, letter, count):
    scores = [0] * count
    try:
        with open(os.path.join(BASEDIR, experiment, 'stability', f'{letter}.json'), 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        return scores
    for trial in data['raws']:
        index = int(re.search(r'(\d+)(?=\.xml$)', trial['tag']).group(1)) - 1
        scores[index] = trial['score']
    return scores

def load_diversity_vectors(experiment, letter, count):
    vectors = [None] * count
    try:
        with open(os.path.join(BASEDIR, experiment, 'diversity', f'{letter}.json'), 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        return vectors
    for trial in data['trials']:
        index = int(re.search(r'(\d+$)', trial['trial']).group(1)) - 1
        vectors[index] = trial['vector']
    return vectors

def compute_pairwise_cosine_distances(v):
    indexed_vectors = [(i, vec) for i, vec in enumerate(v) if vec]
    if not indexed_vectors:
        return {}

    indices, valid_vectors = zip(*indexed_vectors)
    valid_matrix = np.vstack(valid_vectors)

    distance_matrix = cosine_distances(valid_matrix)

    distance_table = {}
    for i, idx_i in enumerate(indices):
        for j, idx_j in enumerate(indices):
            distance_table[(idx_i, idx_j)] = distance_matrix[i][j]

    return distance_table

def load_experiments(experiments, letter):
    num_trials = 0
    similarities = []
    stabilities = []
    diversity_vectors = []
    for experiment in experiments:
        experiment_trials = count_expected(experiment, letter)
        num_trials += experiment_trials
        similarities += load_similarity_scores(experiment, letter, experiment_trials)
        stabilities += load_stability_scores(experiment, letter, experiment_trials)
        diversity_vectors += load_diversity_vectors(experiment, letter, experiment_trials)
    return num_trials, similarities, stabilities, diversity_vectors

def simulate_experiment(experiments, iterations):
    letters_no_diversity = [[0] * iterations for _ in range(26)]
    letters_with_diversity = [[0] * iterations for _ in range(26)]
    summary_similarities = []
    summary_diversities = []
    invalid_count = 0
    for letter_idx, letter in enumerate(string.ascii_uppercase):
        num_trials, similarities, stabilities, diversity_vectors = load_experiments(experiments, letter)
        invalid_count += len([x for x in diversity_vectors if not x])
        distance_table = compute_pairwise_cosine_distances(diversity_vectors)

        for i in range(iterations):
            selection = random.choices(range(num_trials), k=10)
            selection = [x for x in selection if diversity_vectors[x]]

            diversity_score = 0
            for pair in itertools.combinations(selection, 2):
                diversity_score += distance_table[pair]
            if len(selection) > 1:
                diversity_score /= 45
                summary_similarities.append(sum(similarities[x] for x in selection) / len(selection))
                summary_diversities.append(diversity_score)

            score_excluding_diversity = sum(similarities[x] * stabilities[x] for x in selection) / 10
            score_including_diversity = score_excluding_diversity * diversity_score

            letters_no_diversity[letter_idx][i] = score_excluding_diversity
            letters_with_diversity[letter_idx][i] = score_including_diversity
    return invalid_count, np.average(np.array(letters_no_diversity), axis=0), np.average(np.array(letters_with_diversity), axis=0), summary_similarities, summary_diversities

def plot_results(scores, baseline, title, file):
    indices = list(range(len(EXPERIMENTS)))
    indices.sort(key=lambda i: np.median(scores[i]), reverse=True)
    data = [scores[i] for i in indices]
    fig, ax = plt.subplots(figsize=(3.5, 3))
    bp = plt.boxplot(data, vert=False, flierprops={'marker': '.', 'markersize': 5})
    for fly in bp['fliers']:
        fdata = fly.get_data()
        fly.set_data([fdata[0][0], fdata[0][-1]], [fdata[1][0], fdata[1][-1]])
    plt.yticks(ticks=range(1, len(indices) + 1), labels=[EXPERIMENTS[i][0] for i in indices])
    plt.axvline(x=baseline, color='r', linestyle='--', linewidth=1, label='2024 Winner')
    plt.xlabel('Prompt Score')
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(file)
    plt.show()

def plot_similarity_vs_diversity(summary_similarities, summary_diversities):
    indexes = random.sample(range(len(summary_similarities)), k=10000)
    similarities = [summary_similarities[x] for x in indexes]
    diversities = [summary_diversities[x] for x in indexes]
    colors = [similarities[i] * diversities[i] for i in range(len(similarities))]

    plt.figure(figsize=(3.5, 2))
    scatter = plt.scatter(similarities, diversities, c=colors, cmap='viridis')
    plt.colorbar(scatter, label='Score')

    plt.xlabel('Similarity')
    plt.ylabel('Diversity')
    plt.title('Individual Character Similarity vs Diversity')

    plt.tight_layout()
    plt.savefig('similarity_vs_diversity.png')
    plt.show()


if __name__ == '__main__':
    scores_excluding_diversity = []
    scores_including_diversity = []
    summary_similarities = []
    summary_diversities = []
    for name, experiments in EXPERIMENTS:
        invalid_count, sed, sid, sum_sim, sum_div = simulate_experiment(experiments, 10000)
        scores_excluding_diversity.append(sed)
        scores_including_diversity.append(sid)
        summary_similarities.extend(sum_sim)
        summary_diversities.extend(sum_div)
        print(name, np.min(sid), '-|', np.mean(sid), '|-', np.max(sid))
        print('\t', np.std(sid), '--', np.var(sid))
        print('Invalid', invalid_count)

    plot_results(scores_including_diversity, 0.09459, 'Prompt Performance Including Diversity', 'results_including_diversity.eps')
    plot_results(scores_excluding_diversity, 0.42891, 'Prompt Performance Excluding Diversity', 'results_excluding_diversity.eps')
    plot_similarity_vs_diversity(summary_similarities, summary_diversities)

