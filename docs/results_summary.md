# Final Results Summary

## What Was Run

- Model: `Qwen/Qwen2.5-3B-Instruct`, revision
  `aa8e72537993ba99e69dfaafa59ed015b17504d1`.
- Precision conditions: unquantized BF16 and 4-bit NF4 weights with BF16
  computation and double quantization disabled.
- Dataset: all 872 labeled SST-2 validation examples.
- Shot counts: `0`, `1`, `2`, `4`, and `8`.
- Demonstrations: six fixed nested selections for every nonzero shot count.
- Total predictions: 43,600.
- Metrics: accuracy and ECE with 10 equal-width confidence bins.
- Uncertainty: 1,000-sample paired hierarchical bootstrap with seed `42` and
  95% percentile intervals.

## Aggregate Metrics

Nonzero-shot values are means across six demonstration selections. The value
after `+/-` is the sample standard deviation across those selections.

| Precision | Shots | Accuracy | ECE |
|---|---:|---:|---:|
| BF16 | 0 | 0.8773 | 0.1066 |
| BF16 | 1 | 0.8442 +/- 0.0681 | 0.1264 +/- 0.0578 |
| BF16 | 2 | 0.8947 +/- 0.0317 | 0.0879 +/- 0.0258 |
| BF16 | 4 | 0.9256 +/- 0.0087 | 0.0627 +/- 0.0056 |
| BF16 | 8 | 0.9251 +/- 0.0068 | 0.0636 +/- 0.0057 |
| 4-bit NF4 | 0 | 0.9014 | 0.0835 |
| 4-bit NF4 | 1 | 0.8628 +/- 0.0962 | 0.1045 +/- 0.0820 |
| 4-bit NF4 | 2 | 0.9117 +/- 0.0233 | 0.0664 +/- 0.0158 |
| 4-bit NF4 | 4 | 0.9335 +/- 0.0129 | 0.0536 +/- 0.0104 |
| 4-bit NF4 | 8 | 0.9287 +/- 0.0066 | 0.0568 +/- 0.0068 |

## Paired Quantization Gaps

Every gap is `4-bit - BF16`. Positive accuracy gaps favor 4-bit; negative ECE
gaps indicate lower ECE for 4-bit.

| Shots | Accuracy gap (95% CI) | ECE gap (95% CI) |
|---:|---:|---:|
| 0 | +0.0241 [0.0115, 0.0367] | -0.0232 [-0.0364, -0.0102] |
| 1 | +0.0185 [-0.0124, 0.0528] | -0.0219 [-0.0498, 0.0037] |
| 2 | +0.0170 [0.0069, 0.0283] | -0.0215 [-0.0319, -0.0106] |
| 4 | +0.0078 [-0.0038, 0.0206] | -0.0091 [-0.0190, 0.0003] |
| 8 | +0.0036 [-0.0042, 0.0111] | -0.0067 [-0.0133, -0.0002] |

## Answer to the Research Question

The observed 4-bit model was more accurate and had lower ECE at every shot
count, but these advantages narrowed as demonstrations were added. The direct
interaction compares each shot-specific quantization gap with the 0-shot gap.

- At 8 shots, the accuracy interaction was `-0.0205`, with 95% CI
  `[-0.0342, -0.0080]`.
- At 8 shots, the ECE interaction was `+0.0164`, with 95% CI
  `[0.0035, 0.0294]`.

Both 8-shot interaction intervals exclude zero. Under the approved analysis,
this is evidence that quantization changed the effect of moving from zero to
eight demonstrations: quantization's initial accuracy and calibration
advantages became smaller. This does not mean quantization harmed absolute
performance; the 4-bit condition remained descriptively better at eight shots.

The 1-shot results varied substantially across demonstration selections, which
shows why multiple fixed selections and paired uncertainty analysis were
necessary.

## Interpretation Limits

- The percentile intervals are not corrected for multiple comparisons.
- Only six demonstration selections were evaluated.
- ECE depends on the fixed 10-bin estimator.
- The result applies to this checkpoint, SST-2 task, prompt, verbalizers, and
  exact NF4 configuration; it should not automatically be generalized to all
  language models or quantization methods.

Exact machine-readable results are in `results/tables/full_summary.json` and
`results/tables/bootstrap_summary.json`.
