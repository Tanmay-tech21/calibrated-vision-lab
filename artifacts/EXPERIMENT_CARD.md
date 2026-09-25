# Experiment Card: Calibrated Vision Lab

## Claim boundary

This report validates a deterministic calibration-analysis pipeline. It does
not claim performance for a trained vision model or a real deployment.

## Protocol

- Fixture seed: `61`
- Shift seed: `113`
- Calibration split: `1000` samples
- Evaluation split: `2000` untouched samples
- Temperature fitting: clean calibration split only
- ECE convention: equal-width top-label ECE with `15` bins

## Clean held-out results

| Method | Accuracy | NLL | Brier | ECE | AURC |
|---|---:|---:|---:|---:|---:|
| Raw | 0.7825 | 0.9492 | 0.3967 | 0.1440 | 0.1686 |
| Temperature scaled | 0.7825 | 0.9297 | 0.4198 | 0.1764 | 0.1701 |

## Controlled shift results at severity 1.5

| Stressor | Method | Accuracy | NLL | ECE | AURC |
|---|---|---:|---:|---:|---:|
| confidence softening | raw | 0.7825 | 1.0429 | 0.3277 | 0.1747 |
| confidence softening | calibrated | 0.7825 | 1.1140 | 0.3806 | 0.1759 |
| gaussian noise | raw | 0.5775 | 1.3636 | 0.1590 | 0.2918 |
| gaussian noise | calibrated | 0.5775 | 1.2329 | 0.1009 | 0.2923 |
| class bias | raw | 0.6710 | 1.0552 | 0.0622 | 0.2194 |
| class bias | calibrated | 0.6710 | 1.0055 | 0.0544 | 0.2214 |

Severity is internal to each stressor; values are not comparable between
stressor families.

## Interpretation

Temperature scaling improves its clean held-out NLL objective without
guaranteeing improvement in ECE, Brier score, or selective ranking. Under
shift, changes in aggregate ECE can also disagree with changes in accuracy.
Accordingly, no single metric is treated as sufficient evidence of reliability.

## Limitations

- The logits and labels are synthetic software fixtures, not model benchmarks.
- Logit stressors isolate failure modes but do not reproduce image corruptions.
- ECE is bin-dependent and must be interpreted with proper scoring rules.
- No deployment threshold is validated on real coin images.

## Intended next validation

Evaluate genuine held-out model logits under measured lighting, blur, glare,
compression, and background changes. Select abstention thresholds on calibration
data and report realised risk and coverage on a separately preserved test set.
