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

The exact model revision and tokenizer revision must be pinned after the pilot confirms that the model works in the university environment.

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
  revision: null

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
  revision: null

precision:
  condition_name: 4bit_nf4
  quantized: true
  load_in_4bit: true
  quantization_type: nf4
  compute_dtype: bfloat16
  use_double_quantization: false
```

The tokenizer, prompt construction, demonstration selection, evaluation examples, batching procedure, and scoring method must remain identical across both conditions.

The pilot must inspect which model modules are quantized and which remain in higher precision.

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

A paired hierarchical bootstrap with 1,000 samples is planned for the final analysis, but it should only be implemented after the main inference and metric pipeline has passed the pilot.

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

The exact model and tokenizer revisions should be pinned after this pilot.

## Full Experiment

After the pilot succeeds:

1. evaluate BF16 on all 872 validation examples;
2. evaluate 4-bit NF4 on the same examples;
3. run all shot counts;
4. run all six demonstration seeds for non-zero-shot settings;
5. save raw prediction records;
6. compute accuracy and ECE;
7. create reliability diagrams;
8. calculate paired precision gaps;
9. produce final tables and figures.

Do not run the full experiment before validating the pilot outputs.

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

## Planned Implementation Modules

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

### `src/inference.py`

- score verbalizers;
- normalize probabilities;
- create prediction records;
- run batched inference.

### `src/metrics.py`

- accuracy;
- ECE;
- reliability-bin statistics;
- paired metric gaps.

### `src/plots.py`

- reliability diagrams;
- accuracy plots;
- ECE plots;
- final paper figures.

### `scripts/prepare_sst2.py`

- create and save dataset manifests.

### `scripts/run_pilot.py`

- run the small pilot.

### `scripts/run_full_experiment.py`

- run the complete experiment after pilot validation.

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
