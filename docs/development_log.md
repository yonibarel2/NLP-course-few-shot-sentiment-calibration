# Development and Experiment Log

This document records how the implementation was built, validated, and run.
It complements the README, which describes the final system, and
`docs/experiment_spec.md`, which defines the experimental protocol.

## 1. Protocol and Repository Setup

- Defined the research question: whether 4-bit quantization changes how shot
  count affects sentiment-classification calibration.
- Fixed the model to `Qwen/Qwen2.5-3B-Instruct`.
- Fixed the comparison to unquantized BF16 versus 4-bit NF4 weights with BF16
  computation and double quantization disabled.
- Fixed SST-2, the `negative`/`positive` verbalizers, shot counts
  `0, 1, 2, 4, 8`, and demonstration seeds `0, 1, 2, 3, 4, 5`.
- Recorded the rule that prompts, examples, checkpoint, tokenizer, batching,
  and scoring must be identical across precision conditions.
- Added repository hygiene rules excluding model weights, caches, secrets,
  environments, and large raw outputs from Git.

## 2. Dataset Preparation

- Loaded `stanfordnlp/sst2` through Hugging Face Datasets.
- Verified the expected 67,349 training examples and 872 labeled validation
  examples.
- Reserved a deterministic, stratified 200-example prompt-development subset:
  100 negative and 100 positive examples.
- Excluded those 200 examples from the demonstration pool.
- Saved split fingerprints and selected identifiers in
  `data/splits/sst2_split_manifest.json`.
- Added tests for split sizes, balance, disjointness, coverage, and
  deterministic regeneration.

## 3. Majority-Class Baseline

- Computed the baseline using only the SST-2 training split.
- The training majority class was `positive`: 37,569 of 67,349 examples.
- Its fixed confidence was the empirical majority frequency,
  `0.5578256544`.
- On validation, baseline accuracy was `0.5091743119` and 10-bin ECE was
  `0.0486513425`.
- Saved the complete result and reliability bins in
  `results/tables/majority_class_baseline.json`.

## 4. Demonstrations and Prompts

- Generated six deterministic demonstration selections using seeds 0 through
  5.
- Sampled four positive and four negative demonstrations per seed.
- Alternated the first demonstration class across seeds.
- Enforced balanced 2-, 4-, and 8-shot prefixes and nested 1-, 2-, 4-, and
  8-shot conditions.
- Saved demonstration identities and order in
  `data/splits/sst2_demonstration_sets.json`.
- Implemented the fixed instruction, labeled demonstration format, and
  unlabeled evaluation query.
- Added methodological tests for deterministic demonstrations, nested
  prefixes, class balance, and prompt formatting.

## 5. Verbalizer Inspection and Scoring

- Loaded the Qwen tokenizer and inspected both label verbalizers in prompt
  context before implementing model scoring.
- Pinned model and tokenizer revision
  `aa8e72537993ba99e69dfaafa59ed015b17504d1`.
- Verified that both contextual verbalizers are single tokens:
  `negative` token ID `42224` and `positive` token ID `30487`.
- Implemented direct label-score extraction and restricted softmax over the
  two labels; the model never generates free-form classifications or reported
  confidence values.
- Retained a full-sequence summed-log-probability fallback for any future
  multi-token verbalizer.
- Saved negative/positive scores and probabilities, selected-label
  confidence, prediction, correctness, prompt hash, revisions, precision,
  shots, seed, and example identifiers for every prediction.

## 6. GPU Environment

- Used a RunPod PyTorch container with one NVIDIA L40 GPU.
- Recorded driver `550.127.08`, CUDA runtime `12.8`, compute capability `8.9`,
  and BF16 support.
- The validated software environment used Python `3.12.3`, PyTorch
  `2.8.0+cu128`, Transformers `5.13.1`, Datasets `5.0.0`, bitsandbytes
  `0.49.2`, and Accelerate `1.14.0`.
- Cloned the GitHub repository into `/workspace` and installed the declared
  requirements.
- Confirmed CUDA availability and ran the repository tests before inference.

## 7. Pilot and Batching Decision

- Implemented a pilot covering 20 validation examples, 0-shot and 2-shot,
  demonstration seed 0, and both precision conditions.
- The pilot generated and validated 80 predictions.
- Checked successful BF16 and NF4 loading, valid probability sums, prediction
  consistency, identical prompts, required output fields, and manageable
  memory/runtime.
- The first general padded implementation was correct but inefficient for
  mixed prompt lengths.
- Tested equal-token-length grouping and one shared-prefix forward pass for
  the verified single-token labels. Predictions were unchanged, although
  finite-precision probabilities differed slightly because batching changes
  floating-point execution order.
- Explained this implementation choice and obtained approval before making it
  the final batching procedure.
- Fixed batch size 8, identical grouping in both precision conditions, and
  restoration of original deterministic record order.
- Reran the complete pilot successfully with the finalized scorer before
  allowing the full experiment.

## 8. Full Experiment

- Evaluated all 872 validation examples.
- Evaluated 0-shot once per precision condition.
- Evaluated each of 1-, 2-, 4-, and 8-shot with all six demonstration seeds
  under both precision conditions.
- Produced 43,600 total prediction records.
- Saved a checkpoint after each precision condition on RunPod.
- BF16 inference, including model loading, took approximately 239 seconds;
  4-bit NF4 took approximately 308 seconds.
- Validated complete condition coverage, no duplicate condition/example keys,
  probability normalization, prediction consistency, and identical prompt
  hashes across precision conditions.
- Generated condition metrics, seed aggregates, paired gaps, reliability data,
  and 13 initial final figures.
- Copied the combined raw predictions, tables, and figures to the local
  workspace and verified them against the SHA-256 hashes recorded by the full
  run.

## 9. Statistical Analysis

- The experiment specification required a 1,000-sample paired hierarchical
  bootstrap but did not fully define the resampling hierarchy.
- Recommended and obtained approval for a crossed paired method: each
  replicate resamples validation examples once and demonstration seeds once,
  applying the same draws to every precision and shot condition.
- Used deterministic bootstrap seed `42` and 95% percentile intervals.
- Calculated uncertainty for every `4-bit - BF16` accuracy and ECE gap.
- Calculated shot-effect interactions as each nonzero-shot quantization gap
  minus the 0-shot quantization gap.
- Generated exact bootstrap CSV/JSON outputs and the interaction figure.
- The final analysis found that the 4-bit condition was descriptively more
  accurate and had lower ECE at every shot count, while both advantages
  narrowed as shot count increased. The 8-shot-versus-0-shot interactions for
  both metrics had intervals excluding zero.
- Documented limitations: six demonstration selections, fixed 10-bin ECE,
  percentile intervals without multiple-comparison correction, and results
  specific to this model/task/configuration.

## 10. Final Validation and Version Control

- Updated the README and experiment specification from planned to completed
  status.
- Added the final result interpretation in `docs/results_summary.md`.
- Added line-ending rules so committed machine-readable results retain stable
  hashes across Windows and Linux.
- Kept the 30.9 MB combined raw prediction JSONL outside Git according to the
  repository hygiene policy.
- Committed code, configuration, aggregate tables, and all 14 figures in commit
  `c5b49e2` and pushed it to GitHub.
- Tested that exact pushed commit in an isolated RunPod Git worktree: all 29
  tests passed.

## 11. Artifact Locations

- Combined raw predictions, retained locally but ignored by Git:
  `results/raw/full_predictions.jsonl`.
- Full-run metadata and hashes: `results/tables/full_summary.json`.
- Exact aggregate metrics: `results/tables/full_aggregate_metrics.csv`.
- Bootstrap method, validation, and hashes:
  `results/tables/bootstrap_summary.json`.
- Bootstrap intervals: `results/tables/bootstrap_gap_intervals.csv` and
  `results/tables/bootstrap_interaction_intervals.csv`.
- Human-readable result interpretation: `docs/results_summary.md`.
- Final figures: `results/figures/`.

## 12. RunPod Artifacts Not Needed Locally

The following Pod artifacts are deliberately not part of the submission:

- downloaded Hugging Face model weights and caches;
- the installed Python/container environment;
- temporary files and test caches;
- redundant per-precision raw checkpoints, because the hash-verified combined
  raw file contains both complete conditions.

GPU VRAM and system RAM are temporary working memory and contain nothing that
needs to be copied. The essential persistent experiment outputs are already in
the local workspace.

## 13. Retained Module-Precision Audit

- During report preparation, identified that the pilot had verified successful
  BF16 and 4-bit loading/inference but had not retained the module-by-module
  datatype inventory requested by the specification.
- Explained the gap and obtained approval for a short structural audit. This
  was not a rerun of SST-2 inference and did not modify any experimental result.
- The original stopped L40 host had no available GPU, so an A100-SXM4-80GB
  temporary Pod was used. Its attached network volume was empty; the pushed
  repository was cloned and the exact full-run PyTorch `2.8.0+cu128`,
  Transformers `5.13.1`, Accelerate `1.14.0`, and bitsandbytes `0.49.2`
  versions were recreated.
- Loaded revision `aa8e72537993ba99e69dfaafa59ed015b17504d1` sequentially
  under the approved BF16 and 4-bit configurations.
- Confirmed that the BF16 model contained 434 BF16 parameter tensors and no
  detected quantized weight modules.
- Confirmed that the quantized model contained 252 NF4 `Linear4bit` weight
  modules stored in packed `uint8` tensors with BF16 computation. The other 182
  parameter tensors remained BF16, including embeddings, normalization
  parameters, and the output head. Double quantization was disabled.
- The reported model footprints were approximately 5.75 GiB for BF16 and
  1.87 GiB for 4-bit.
- Saved the complete audit to
  `results/tables/model_precision_inventory.json`, copied it locally, and
  verified SHA-256
  `5a5f8d73324afca33cc9d8acba9ee77a254c553d213742ee11b034afc9dfa7c0`
  against the Pod copy.
- Ran the two audit-specific tests on the GPU environment; both passed.
