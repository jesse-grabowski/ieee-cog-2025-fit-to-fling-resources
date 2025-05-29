# Fit to Fling: Supplemental Data

This repository includes supplemental code and data for the Fit to Fling competition paper. The contents are described below.

## ablation_data

The results of our ablation trials for Phi3 on one seed (24). This data was also used to populate our results for Section V.

As our full dataset is 12GB, it won't fit in Anonymous GitHub. This repository includes a representative sample; after the paper is reviewed we will publish the entire dataset on GitHub.

This data is in the LLMs4PCG competition script format, so any tooling that works there should work on it as well.

## baseline / decode / diversify / recall

The results of automatic prompt optimization for the four tasks described in our paper. Each directory follows the format:

```
<name>/
├─ <generation>/
│  ├─ meta.json
│  ├─ <uuid>.txt
├─ evaluation-criteria.txt
```

* `<name>` - the name of the task
* `<generation>` - generation number, 0 represents seeds
* `meta.json` - contains information about each prompt's evaluation performance and lineage, described below
* `<uuid>.txt` - a prompt
* `evaluation-criteria.txt` - criteria used during optimization, see `optimize.py`

`meta.json` follows the format:

```
{
	"<prompt uuid>": {
		"score": 0 // summary evaluation score
		"parents": ["<uuid>", ...] // 0-2 parents that created this prompt
		"suggestion": "e.g. crossover" // the mutation instruction that created this prompt
		"criteria_scores": [0, 1, ...] // individual test case performance, order matches evaluation-criteria
		"responses": ["lorem", "ipsum", ...] // the output of the evaluation model
	}
}
```

## monte_carlo.py

Our simulation script - utilizes data from `ablation_data`

## optimize.py

Our automatic prompt optimization script. Expects a `.env` file with the following contents:

```
OPENAI_API_KEY=<key>
```

Additionally, expects a file named `prompts/0/meta.json` and at least one seed prompt file to exist in the format described above.

Make sure Ollama is running and that your evaluation model has already been downloaded.

## requirements.txt

Python dependency versions