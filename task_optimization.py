import os
import json
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

from matplotlib.ticker import MaxNLocator

plt.style.use(['science','ieee'])

def load_scores_from_meta(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    scores = [entry["score"] for entry in data.values()]
    scores = list(filter(lambda x: x >= 0, scores))
    return scores

def gen_stats(base_dir, ax, title):
    generations = sorted([d for d in os.listdir(base_dir) if d.isdigit()], key=int)

    avg_scores = []
    std_scores = []
    max_scores = []
    gen_labels = []

    for gen in generations:
        meta_path = os.path.join(base_dir, gen, "meta.json")
        if os.path.isfile(meta_path):
            scores = load_scores_from_meta(meta_path)
            if scores:
                avg_scores.append(np.mean(scores))
                std_scores.append(np.std(scores))
                max_scores.append(np.max(scores))
                gen_labels.append(int(gen))

    gen_array = np.array(gen_labels)
    avg_array = np.array(avg_scores)
    std_array = np.array(std_scores)
    max_array = np.array(max_scores)

    ax.plot(gen_array, max_array, label="Max Fitness", marker=".")
    ax.plot(gen_array, avg_array, label="Average Fitness", color="#54c45e")
    ax.fill_between(gen_array, avg_array - std_array, avg_array + std_array,
                     color='#f2f2ff', label="±1 Std Dev")

    ax.set_xlabel("Generation")
    ax.set_xlim(min(gen_array), max(gen_array))
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel("Fitness")
    ax.set_ylim(0, 1.1)
    ax.set_title(title)
    ax.legend()


if __name__ == "__main__":
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(5, 4))
    gen_stats('diversify', axes[0][0], 'Diversify')
    gen_stats('recall', axes[0][1], 'Recall')
    gen_stats('decode', axes[1][0], 'Decode')
    gen_stats('baseline', axes[1][1], 'Baseline')
    plt.tight_layout()
    plt.savefig("fitness_combined.eps")
    plt.show()