# Quantized Few-Shot Calibration

This repository contains the code for the NLP final project:

**Does Quantization Change Few-Shot Prompting's Effect on Calibration?**

## Project Overview

This project studies whether 4-bit quantization changes how few-shot prompting affects a language model's accuracy and calibration.

We evaluate the same instruction-tuned language model in two conditions:

1. Higher-precision inference
2. 4-bit quantized inference

The model performs binary sentiment classification on SST-2 using:

- 0-shot prompting
- 1-shot prompting
- 2-shot prompting
- 4-shot prompting
- 8-shot prompting

For each prediction, we record:

- the true label;
- the predicted label;
- the probability assigned to each label;
- the model's confidence;
- whether the prediction was correct.

We compare the two precision conditions using accuracy, Expected Calibration Error (ECE), and reliability diagrams.

## Research Question

Does 4-bit quantization change the effect of increasing the number of in-context demonstrations on model calibration?

More specifically, we examine whether quantization changes:

- classification accuracy;
- confidence reliability;
- the relationship between shot count and calibration.

## Dataset

We use SST-2, the binary version of the Stanford Sentiment Treebank.

Each example contains a movie-review sentence labeled as:

- `negative`
- `positive`

The training split is used to construct:

- a fixed prompt-development subset;
- a demonstration pool for few-shot examples.

The labeled validation split is used as the final evaluation set.

Evaluation examples are never used as demonstrations or for prompt development.

## Experimental Design

The experiment compares:

| Condition | Values |
|---|---|
| Model | `Qwen/Qwen2.5-3B-Instruct` |
| Precision | BF16, 4-bit NF4 |
| Shot count | 0, 1, 2, 4, 8 |
| Task | Binary sentiment classification |
| Labels | `positive`, `negative` |
| Demonstration seeds | 0, 1, 2, 3, 4, 5 |

The same model checkpoint, tokenizer, prompts, demonstrations, evaluation examples, batching procedure, and scoring method are used for both precision conditions.

The only intended experimental difference is the numerical representation of the model weights.

## Model

The experiment uses one model checkpoint:

```text
Qwen/Qwen2.5-3B-Instruct
```

The same checkpoint is evaluated in two conditions:

1. Higher precision – unquantized BF16 inference
2. Quantized – 4-bit NF4 weights with BF16 computation and double quantization disabled

The model, tokenizer, prompts, demonstrations, evaluation examples, and scoring procedure are identical across both conditions.

The pilot pinned both the model and tokenizer to revision
`aa8e72537993ba99e69dfaafa59ed015b17504d1`. Exact package versions, GPU
configuration, runtimes, and peak memory are recorded in
`results/tables/pilot_summary.json`.

## Precision Configurations

### Higher Precision

Configuration file:

```text
configs/high_precision.yaml
```

Main settings:

```yaml
model:
  name: Qwen/Qwen2.5-3B-Instruct
  revision: aa8e72537993ba99e69dfaafa59ed015b17504d1

precision:
  condition_name: bf16
  quantized: false
  torch_dtype: bfloat16
```

### 4-bit Quantized

Configuration file:

```text
configs/quantized_4bit.yaml
```

Main settings:

```yaml
model:
  name: Qwen/Qwen2.5-3B-Instruct
  revision: aa8e72537993ba99e69dfaafa59ed015b17504d1

precision:
  condition_name: 4bit_nf4
  quantized: true
  load_in_4bit: true
  quantization_type: nf4
  compute_dtype: bfloat16
  use_double_quantization: false
```

Both configurations use the exact revision pinned by the successful pilot.

The retained module-level precision audit is stored in
`results/tables/model_precision_inventory.json`. It found no quantized
weight modules in the BF16 load. In the 4-bit load, 252 weight modules were
bitsandbytes `Linear4bit` modules with NF4 weights and BF16 computation;
embeddings, normalization parameters, and the output head remained BF16. The
recorded bitsandbytes configuration also confirms that double quantization was
disabled. This structural audit did not perform inference or alter the full-run
predictions.

## SST-2 Data Protocol

A fixed stratified subset of 200 training examples is reserved for prompt development and debugging:

- 100 negative examples
- 100 positive examples

These examples are excluded from the demonstration pool.

The demonstration pool contains all remaining training examples.

The full labeled validation split is used as the final evaluation set.

The validation labels are not used for:

- prompt development;
- demonstration selection;
- model selection;
- configuration tuning.

### Data Preparation

The deterministic partition settings are stored in:

```text
configs/data.yaml
```

Prepare and validate the SST-2 split manifest with:

```text
python scripts/prepare_sst2.py
```

The preparation step uses seed `42`, reserves 100 training examples per
class for prompt development, and saves the selected identifiers and dataset
fingerprints to `data/splits/sst2_split_manifest.json`. It validates source
split sizes, validation class counts, prompt-development balance, training
partition disjointness, and complete training-set coverage without committing
a full dataset copy.

## Demonstration Construction

The experiment uses shot counts:

```text
0, 1, 2, 4, 8
```

For each seed, eight demonstrations are sampled from the demonstration pool:

- four positive;
- four negative.

The demonstration sets are nested. For a given seed:

- the 1-shot prompt uses the first demonstration;
- the 2-shot prompt uses the first two;
- the 4-shot prompt uses the first four;
- the 8-shot prompt uses all eight.

The 2-, 4-, and 8-shot prefixes are class-balanced.

The first demonstration's label alternates across seeds so that the 1-shot condition does not consistently favor one class.

The same demonstration identities and ordering are used in both precision conditions.

Generate and validate all six selections with:

```text
python scripts/generate_demonstrations.py
```

The ordered identifiers and labels are saved in
`data/splits/sst2_demonstration_sets.json`. The manifest records the source
split-manifest hash and dataset fingerprint, and the generator asserts that
prompt-development examples are excluded.

## Prompt Format

Each prompt contains:

1. a task instruction;
2. zero or more labeled demonstrations;
3. one unlabeled evaluation sentence.

Example:

```text
Classify the sentiment of each movie-review sentence as positive or negative.

Review: The movie was funny and beautifully acted.
Sentiment: positive

Review: The plot was dull and predictable.
Sentiment:
```

The label verbalizers are:

```text
positive
negative
```

Before the full experiment, we verified the tokenizer representation of both
verbalizers. The scorer uses summed conditional log-probabilities for the
complete token sequence whenever a verbalizer contains multiple tokens.

The exact plain-text prompt construction and saved-prefix materialization are
implemented in `src/prompts.py`. Tokenizer and chat-template handling remain a
separate validation step because they depend on the selected model tokenizer.

Inspect the verbalizers in real 0-shot and 2-shot prompt contexts with:

```text
python scripts/inspect_verbalizers.py
```

The validated report is saved to
`results/tables/verbalizer_tokenization.json`. With the Qwen tokenizer at
resolved revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`, both contexts
produce the same single-token continuations:

- `negative`: token ID `42224`;
- `positive`: token ID `30487`.

The implementation still records complete token sequences so that a future
tokenizer change cannot silently introduce an incorrect single-token
assumption. The successful pilot subsequently pinned the model and tokenizer
configuration to that inspected revision.

## Confidence Extraction

The model does not generate a free-form answer or a self-reported confidence value.

Instead, we obtain model scores for the two label verbalizers:

- `positive`
- `negative`

The two scores are normalized using a restricted softmax.

The label with the higher probability is selected as the prediction.

The probability assigned to the selected label is used as the prediction confidence.

## Evaluation Metrics

### Accuracy

Accuracy is the proportion of evaluation examples classified correctly.

### Expected Calibration Error

Expected Calibration Error measures the difference between prediction confidence and observed accuracy.

Predictions are grouped into 10 equal-width confidence bins.

A lower ECE indicates better calibration.

### Reliability Diagrams

Reliability diagrams compare:

- mean confidence;
- empirical accuracy.

A well-calibrated model should remain close to the diagonal line.

## Majority-Class Baseline

The majority-class baseline always predicts the most frequent label in the SST-2 training split.

Its confidence is set to the empirical training frequency of that label.

The majority label and confidence are computed using the training split only.

Compute the baseline with:

```text
python scripts/compute_majority_baseline.py
```

The validated result is stored in
`results/tables/majority_class_baseline.json`, including all 10 equal-width
reliability bins. On the recorded SST-2 splits, the training majority label is
`positive` (37,569 of 67,349 examples), giving a fixed confidence of
`0.5578256544`. On the complete validation split, its accuracy is
`0.5091743119` and its ECE is `0.0486513425`. Validation labels are used only
to evaluate accuracy and calibration; they do not determine the prediction or
confidence.

## Tools

The project uses:

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- bitsandbytes
- NumPy
- pandas
- Matplotlib
- PyYAML
- pytest

No paid language-model API is required.

## Repository Structure

```text
quantized-few-shot-calibration/
├── README.md                 # design, commands, results, and reproduction
├── requirements.txt         # recorded runtime and analysis versions
├── configs/                 # data, precision, and analysis settings
├── data/splits/             # deterministic split and demonstration manifests
├── docs/                    # experiment specification and result summaries
├── report/                  # ACL LaTeX source and publication-quality figures
├── results/
│   ├── figures/             # generated analysis figures
│   └── tables/              # metrics, intervals, and run metadata
├── scripts/                 # preparation, inference, analysis, and validation CLIs
├── src/                     # data, prompting, inference, metrics, and reporting code
└── tests/                   # deterministic and methodological tests
```

Large raw predictions are deliberately retained outside Git under
`results/raw/`. Local course documents and working instructions are also
excluded from the submitted repository.

## Experimental Pipeline

1. Load SST-2.
2. Reserve the balanced prompt-development subset.
3. Create the demonstration pool.
4. Define the validation evaluation set.
5. Compute the majority-class baseline.
6. Create fixed demonstration selections for seeds 0 through 5.
7. Build prompts for 0, 1, 2, 4, and 8 shots.
8. Verify label-token verbalization.
9. Load the model in BF16.
10. Load the same model using 4-bit NF4 quantization.
11. Extract label probabilities for every evaluation example.
12. Save predictions and confidence values.
13. Compute accuracy and ECE.
14. Generate reliability diagrams, tables, and figures.
15. Compare how shot count affects both precision conditions.

## Pilot Experiment

Before the full experiment, the fixed pilot used:

- approximately 20 SST-2 examples;
- 0-shot and 2-shot prompts;
- one demonstration selection;
- both precision conditions.

Run the fixed pilot with:

```text
python scripts/run_pilot.py
```

The runner uses the first 20 validation examples without tuning, evaluates
0-shot and 2-shot prompts with demonstration seed 0, and loads the exact same
checkpoint first in BF16 and then in 4-bit NF4. It scores complete verbalizer
sequences from model logits, applies a restricted two-label softmax, writes
the required per-prediction fields to
`results/raw/pilot_predictions.jsonl`, and saves validation metadata to
`results/tables/pilot_summary.json`. The script fails unless probabilities,
predictions, prompt identity across precision conditions, dataset
fingerprints, model/tokenizer revisions, and the approved quantization
configuration all validate.

Because both inspected Qwen verbalizers contain one token, the validated
scorer groups prompts by equal tokenized length and obtains both label scores
from one shared-prefix forward pass. Both precision conditions use the same
grouping and batch size (`8`). A general full-sequence fallback remains in
place for any future multi-token verbalizer.

The pilot verified that:

- the model loads successfully;
- BF16 inference works;
- 4-bit quantization works;
- label tokenization is handled correctly;
- confidence extraction is correct;
- predictions are saved in the expected format;
- both conditions use identical prompts and examples.

All preparation and tokenizer-inspection steps are CPU-only. A BF16-capable GPU
is needed only for loading the model and running the two inference conditions.
The pilot recorded exact GPU, CUDA, runtime, and memory information before the
full experiment was attempted.

The pilot passed on an NVIDIA L40 using CUDA 12.8 and PyTorch 2.8.0. It
produced 80 validated predictions: 20 examples times two shot conditions times
two precision conditions. BF16 peak allocated GPU memory was approximately
6.22 GB and 4-bit peak allocated memory was approximately 2.28 GB. Pilot
metrics are stored only as pipeline checks and are not treated as final
experimental results.

The full experiment completed successfully after the pilot passed.

Run the complete paired experiment with:

```text
python scripts/run_full_experiment.py
```

The full runner evaluates 43,600 predictions: 872 validation examples under
both precision conditions, with 0-shot evaluated once and each nonzero shot
count evaluated for all six demonstration seeds. It writes condition-level
accuracy/ECE, means and sample standard deviations across seeds, paired
4-bit-minus-BF16 gaps, pooled reliability-bin data for visualization, final
CSV tables, and accuracy, ECE, gap, and reliability figures. Per-condition
metrics remain separate; seed pooling is used only for reliability diagrams.

Run the approved 1,000-sample paired hierarchical bootstrap with:

```text
python scripts/run_bootstrap_analysis.py
```

The bootstrap uses seed `42` and 95% percentile intervals. It resamples the
evaluation examples and demonstration selections with replacement, using the
same draws across both precision conditions and all shot counts. This preserves
the paired design. It also estimates the interaction between quantization and
adding `1`, `2`, `4`, or `8` demonstrations relative to 0-shot.

## Final Results

The full experiment and bootstrap both passed all programmed validation
checks. The principal descriptive results are:

| Shots | BF16 accuracy | 4-bit accuracy | BF16 ECE | 4-bit ECE |
|---:|---:|---:|---:|---:|
| 0 | 0.8773 | 0.9014 | 0.1066 | 0.0835 |
| 1 | 0.8442 | 0.8628 | 0.1264 | 0.1045 |
| 2 | 0.8947 | 0.9117 | 0.0879 | 0.0664 |
| 4 | 0.9256 | 0.9335 | 0.0627 | 0.0536 |
| 8 | 0.9251 | 0.9287 | 0.0636 | 0.0568 |

Nonzero-shot entries are means across the six demonstration selections. In
this experiment, 4-bit NF4 had higher mean accuracy and lower mean ECE at every
shot count. However, both advantages became smaller as shot count increased.
Relative to 0-shot, the 8-shot quantization interaction was `-0.0205` for
accuracy (95% CI `[-0.0342, -0.0080]`) and `+0.0164` for ECE (95% CI
`[0.0035, 0.0294]`). These intervals provide evidence that quantization changed
the effect of adding eight demonstrations: its initial advantage narrowed at
eight shots.

These are 95% percentile intervals without multiple-comparison correction.
Interpretation is limited to this model, dataset, quantization configuration,
prompt, six demonstration selections, and fixed 10-bin ECE estimator. See
[`docs/results_summary.md`](docs/results_summary.md) for the complete concise
interpretation and [`results/tables/bootstrap_summary.json`](results/tables/bootstrap_summary.json)
for exact values.

For a chronological record of implementation decisions, pilot issues,
validation steps, the GPU run, and artifact transfer, see
[`docs/development_log.md`](docs/development_log.md).

## Output Format

Each prediction is saved with the following fields:

```text
model_name
model_revision
tokenizer_revision
precision_condition
shot_count
demonstration_seed
demonstration_identifiers
evaluation_example_identifier
gold_label
predicted_label
negative_log_score
positive_log_score
negative_probability
positive_probability
selected_label_confidence
correctness
prompt_sha256
```

The final repository contains:

- aggregated result tables;
- accuracy figures;
- ECE figures;
- reliability diagrams;
- experimental configuration files;
- reproducibility instructions.

## Reproducibility

The completed experiment records:

- exact model revision;
- exact tokenizer revision;
- package versions;
- GPU type;
- precision configuration;
- demonstration identities and ordering;
- random seeds;
- prompts;
- evaluation-example identifiers;
- raw prediction outputs.

Model weights, Hugging Face caches, virtual environments, secrets, and large
temporary files are not committed to the repository.
The 30.9 MB raw prediction JSONL is hash-verified and retained outside Git
under `results/raw/`; it can be regenerated with the documented full-run
command. Aggregated tables and figures are committed.

The validated full run used an NVIDIA L40 (46 GB), driver `550.127.08`, CUDA
runtime `12.8`, Python `3.12.3`, PyTorch `2.8.0+cu128`, Transformers `5.13.1`,
Datasets `5.0.0`, and bitsandbytes `0.49.2`. Exact metadata and input/output
hashes are stored in `results/tables/full_summary.json`. To reproduce from a
fresh checkout, install `requirements.txt`, run the data, demonstration, and
verbalizer commands earlier in this README, then run the pilot, full experiment,
and bootstrap commands in that order. The pilot and full experiment require a
BF16-capable NVIDIA GPU; data preparation, metrics, and bootstrap analysis are
CPU-only.

Run the optional retained module-level precision audit with:

```text
python scripts/inspect_model_precision.py
```

Because the original stopped L40 instance was unavailable, the follow-up audit
was executed on an NVIDIA A100-SXM4-80GB using the same pinned checkpoint and
the same PyTorch, Transformers, Accelerate, and bitsandbytes versions as the
full run. Hardware performance was not measured and no experimental result was
recomputed during this audit.

## Status

- [x] Research question defined
- [x] Experimental design defined
- [x] ACL paper skeleton prepared
- [x] Final ACL report source and Overleaf package prepared
- [ ] Final compiled PDF visually reviewed in Overleaf
- [x] Model selected
- [x] Higher-precision configuration defined
- [x] 4-bit configuration defined
- [x] Model revision pinned
- [x] SST-2 data preparation implemented
- [x] Majority-class baseline computed
- [x] Demonstration sets generated
- [x] Prompt construction implemented
- [x] Label tokenization validated
- [x] Pilot pipeline implemented
- [x] Pilot experiment completed
- [x] BF16/NF4 module precision boundary audited
- [x] Full experiment pipeline implemented
- [x] Full experiment completed
- [x] Results analyzed with paired hierarchical bootstrap
- [x] Final tables and figures generated
- [x] Reproducibility instructions finalized
