# Experimental Specification

## Project Title

**Does Quantization Change Few-Shot Prompting's Effect on Calibration?**

## Research Question

Does 4-bit quantization change how increasing the number of in-context demonstrations affects model calibration?

The project separately examines:

- the effect of shot count on classification accuracy;
- the effect of shot count on calibration;
- whether these effects differ between BF16 and 4-bit inference.

The main comparison is not between two different models. It is between two numerical-precision configurations of the same model checkpoint.

## Task

The task is binary sentiment classification.

Each English movie-review sentence must be classified as:

- `negative`
- `positive`

## Dataset

Use the Hugging Face dataset:

`stanfordnlp/sst2`

The dataset fields include:

- `sentence`;
- `label`;
- `idx`.

Label mapping:

- `0` = `negative`
- `1` = `positive`

The training split contains 67,349 examples.

The labeled validation split contains 872 examples:

- 428 negative;
- 444 positive.

The public test labels are hidden, so the validation split is used as the final evaluation set.

## Dataset Partitioning

### Prompt-Development Set

Reserve a fixed stratified subset of 200 training examples:

- 100 negative;
- 100 positive.

This subset is used only for:

- prompt development;
- implementation checks;
- debugging;
- sanity tests.

These examples are excluded from the demonstration pool.

### Demonstration Pool

The demonstration pool contains all training examples not included in the 200-example prompt-development set.

All few-shot demonstrations are sampled from this pool.

### Evaluation Set

Use all 872 validation examples as the final evaluation set.

Validation examples and labels must not influence:

- prompt development;
- demonstration selection;
- model selection;
- quantization settings;
- scoring decisions;
- debugging decisions.

The exact split seed and selected identifiers must be saved in a reproducibility manifest.

## Model

Use:

`Qwen/Qwen2.5-3B-Instruct`

This is the only model used in the main experiment.

The pilot pinned both the model and tokenizer to revision
`aa8e72537993ba99e69dfaafa59ed015b17504d1`. The pilot and full experiment
used that exact revision.

## Precision Conditions

### Higher-Precision Condition

Use:

- unquantized model weights;
- BF16 inference;
- the original model checkpoint.

Configuration file:

`configs/high_precision.yaml`

Expected configuration:

```yaml
model:
  name: Qwen/Qwen2.5-3B-Instruct
  revision: aa8e72537993ba99e69dfaafa59ed015b17504d1

precision:
  condition_name: bf16
  quantized: false
  torch_dtype: bfloat16
```

### 4-bit Quantized Condition

Use:

- the same model checkpoint;
- 4-bit NF4 weight quantization;
- BF16 computation;
- double quantization disabled;
- bitsandbytes for model loading.

Configuration file:

`configs/quantized_4bit.yaml`

Expected configuration:

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

The tokenizer, prompt construction, demonstration selection, evaluation examples, batching procedure, and scoring method must remain identical across both conditions.

The pilot was required to inspect which model modules are quantized and which
remain in higher precision. Although the successful pilot verified both model
loads and inference conditions, it did not retain a module-by-module datatype
inventory. This documentation gap was closed with an approved post-experiment
structural audit using the same pinned checkpoint, configurations, and core
software versions. The audit found zero quantized weight modules in BF16 and
252 NF4 `Linear4bit` weight modules with BF16 computation in the quantized
condition. Non-quantized components, including embeddings, normalization
parameters, and the output head, remained BF16. The audit performed no
inference and did not change any predictions, metrics, or conclusions. Exact
module records and hardware metadata are retained in
`results/tables/model_precision_inventory.json`.

## Shot Counts

Evaluate:

`0, 1, 2, 4, 8`

The 0-shot condition contains only the task instruction and evaluation query.

The other conditions contain labeled demonstrations before the evaluation query.

## Demonstration Seeds

Use six demonstration seeds:

`0, 1, 2, 3, 4, 5`

For each seed:

1. sample four positive demonstrations;
2. sample four negative demonstrations;
3. create one ordered list of eight demonstrations;
4. use prefixes of that list for all shot counts.

The ordering must ensure:

- the first two demonstrations contain one example from each class;
- the first four contain two examples from each class;
- all eight contain four examples from each class;
- the first demonstration is positive for half the seeds and negative for half the seeds.

This avoids systematically favoring one class in the 1-shot condition.

The demonstration sets are nested. Increasing the shot count adds demonstrations without replacing earlier ones.

The exact same ordered demonstration list must be used in both precision conditions.

The demonstration identifiers and labels must be saved.

## Prompt Format

Base prompt:

```text
Classify the sentiment of each movie-review sentence as positive or negative.

Review: [demonstration sentence]
Sentiment: [demonstration label]

Review: [evaluation sentence]
Sentiment:
```

For 0-shot evaluation, omit the demonstration blocks.

The two label verbalizers are:

- `negative`;
- `positive`.

The exact tokenizer representation of the verbalizers must be inspected after applying the final prompt and chat template.

If a verbalizer consists of one token, use its logit at the answer position.

If a verbalizer consists of multiple tokens, calculate its score as the sum of the conditional log-probabilities of the full token sequence.

### Final Batching Procedure

The successful pilot established the following fixed batching procedure for
the full experiment:

- tokenize prompts before batching;
- group prompts by equal tokenized sequence length;
- use batch size `8` within each group;
- restore predictions to the original deterministic condition/example order;
- apply exactly the same grouping and batch size in BF16 and 4-bit NF4.

Because both inspected verbalizers contain one token, both label logits are
read from one shared prompt-prefix forward pass. This is exactly the
single-token scoring rule above and does not alter the prompt, label scores,
restricted softmax, or research question. The implementation retains summed
conditional log-probability scoring as a fallback for a multi-token
verbalizer.

## Prediction and Confidence

The model must not generate free-form text for the main classification procedure.

For every evaluation example:

1. construct the prompt;
2. obtain the score for `negative`;
3. obtain the score for `positive`;
4. normalize the two scores using a restricted softmax;
5. select the label with the higher probability;
6. use the selected-label probability as confidence.

The probabilities must satisfy:

`P(negative) + P(positive) approximately equals 1`

Save both probabilities, not only the selected-label confidence.

## Raw Output Schema

Each prediction record should contain:

```text
model_name
model_revision
tokenizer_revision
precision
shot_count
seed
example_id
true_label
predicted_label
negative_probability
positive_probability
confidence
correct
```

Optional useful fields:

```text
prompt
demonstration_ids
input_token_count
runtime_seconds
gpu_memory_usage
```

The exact prompt may be stored separately if storing it in every row creates unnecessary duplication.

## Accuracy

Accuracy is the proportion of evaluation examples for which the predicted label equals the gold label.

Compute accuracy separately for every:

- precision condition;
- shot count;
- demonstration seed.

For 1-, 2-, 4-, and 8-shot settings, report the mean and standard deviation across six demonstration seeds.

The 0-shot condition is evaluated once per precision condition.

## Expected Calibration Error

Use Expected Calibration Error with:

- 10 equal-width confidence bins;
- confidence defined as the probability of the predicted label.

For every non-empty bin:

1. calculate the mean confidence;
2. calculate the empirical accuracy;
3. calculate the absolute difference;
4. weight the difference by the fraction of examples in the bin.

Lower ECE indicates better agreement between confidence and observed accuracy.

Compute ECE separately for every:

- precision condition;
- shot count;
- demonstration seed.

## Reliability Diagrams

Use the same 10 equal-width bins as ECE.

For every non-empty bin, plot:

- x-axis: mean confidence;
- y-axis: empirical accuracy.

Include the diagonal line representing perfect calibration.

Generate reliability diagrams for both precision conditions.

The final presentation may use selected shot counts in the main paper and place additional diagrams in the appendix.

## Majority-Class Baseline

Determine the majority label using the training split only.

The majority-class baseline always predicts that label.

Set its confidence to the empirical training frequency of the majority label.

Evaluate the baseline on the validation set.

Record:

- majority label;
- training frequency;
- validation accuracy;
- ECE;
- reliability-bin statistics.

## Paired Comparison

BF16 and 4-bit results are paired because they use:

- the same model checkpoint;
- the same tokenizer;
- the same demonstrations;
- the same evaluation examples;
- the same prompts;
- the same scoring method.

At each shot count, calculate the quantization gap as:

`4-bit metric - BF16 metric`

For accuracy:

- a negative gap means 4-bit accuracy is lower;
- a positive gap means 4-bit accuracy is higher.

For ECE:

- a positive gap means 4-bit calibration is worse;
- a negative gap means 4-bit calibration is better.

After the pilot and full inference pipeline passed, the final analysis used a
paired hierarchical bootstrap with 1,000 samples, seed `42`, and 95% percentile
intervals. Each replicate resamples the 872 evaluation examples once and the
six demonstration seeds once. The same draws are applied to all precision and
shot conditions. This preserves precision pairing, shared evaluation examples,
and the nested demonstration identities across shot counts.

In addition to the gap at each shot count, the analysis calculates the paired
shot-effect interaction:

`(4-bit - BF16 gap at k shots) - (4-bit - BF16 gap at 0 shots)`

for `k` in `1, 2, 4, 8`. This directly measures whether the effect of adding
demonstrations differs between precision conditions.

## Pilot Experiment

The pilot uses approximately 20 validation examples.

Test:

- 0-shot;
- 2-shot;
- one demonstration seed;
- BF16;
- 4-bit NF4.

The pilot should confirm:

- both configurations load successfully;
- the tokenizer and chat template work;
- the verbalizers are scored correctly;
- probabilities are valid;
- the prediction equals the label with the larger probability;
- prompts are identical across precision conditions;
- output records contain all required fields;
- runtime and memory requirements are manageable.

The pilot passed and pinned the model and tokenizer revisions stated above.

## Full Experiment

The pilot and full experiment completed successfully. The full experiment:

1. evaluate BF16 on all 872 validation examples;
2. evaluate 4-bit NF4 on the same examples;
3. run all shot counts;
4. run all six demonstration seeds for non-zero-shot settings;
5. save raw prediction records;
6. compute accuracy and ECE;
7. create reliability diagrams;
8. calculate paired precision gaps;
9. produce final tables and figures.

The run produced 43,600 validated predictions. Output coverage, probability
normalization, prediction consistency, prompt identity across precision
conditions, and file hashes were checked before analysis. Aggregated results
and bootstrap intervals are recorded under `results/tables/`; final figures
are under `results/figures/`.

## Reproducibility Requirements

Record:

- model name;
- model revision;
- tokenizer revision;
- GPU type;
- CUDA version;
- Python version;
- PyTorch version;
- Transformers version;
- Datasets version;
- bitsandbytes version;
- Accelerate version;
- NumPy version;
- pandas version;
- random seeds;
- split manifest;
- demonstration identifiers and order;
- precision configuration;
- prompt template;
- output schema;
- commands used to run the experiments.

The repository should contain enough information to reproduce all reported tables and figures.

## Implemented Modules

Suggested responsibilities:

### `src/data.py`

- load SST-2;
- create deterministic splits;
- calculate baseline statistics;
- load saved manifests.

### `src/prompts.py`

- generate demonstration sets;
- construct prompts;
- apply label verbalizers;
- verify nested prefixes.

### `src/model.py`

- load the tokenizer;
- load BF16 model;
- load 4-bit model;
- inspect module datatypes.

### `src/precision_inventory.py`

- inventory every weight-bearing module and its stored datatype;
- identify bitsandbytes 4-bit modules;
- verify NF4 weights and BF16 computation;
- validate the intended BF16-versus-quantized module boundary.

### `src/inference.py`

- score verbalizers;
- normalize probabilities;
- create prediction records;
- run batched inference.

### `src/metrics.py`

- accuracy;
- ECE;
- reliability-bin statistics.

### `src/reporting.py`

- aggregate accuracy and ECE across demonstration selections;
- paired metric gaps;
- reliability diagrams and final figures.

### `src/bootstrap.py`

- approved paired hierarchical resampling;
- percentile confidence intervals;
- shot-count interaction estimates.

### `scripts/prepare_sst2.py`

- create and save dataset manifests.

### `scripts/run_pilot.py`

- run the small pilot.

### `scripts/run_full_experiment.py`

- run the complete experiment after pilot validation.

### `scripts/run_bootstrap_analysis.py`

- validate the completed prediction-file hash;
- run the final paired uncertainty analysis;
- generate bootstrap tables and the interaction figure.

## Methodological Constraints

Do not:

- compare different model checkpoints across precision conditions;
- use validation examples as demonstrations;
- use validation labels for prompt development;
- ask the model to report its own confidence;
- use free-form generation as the main prediction method;
- change prompts between BF16 and 4-bit;
- change demonstration selections between conditions;
- silently alter shot counts, seeds, bin counts, or quantization settings;
- report results before validating the saved raw predictions.
