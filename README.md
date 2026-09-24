# Calibrated Vision Lab

A seven-day deep-learning laboratory for measuring and improving confidence
calibration in image classifiers. The project begins with framework-independent
NumPy metrics, then builds towards post-hoc calibration and reliability
analysis on real model outputs.

## Why calibration matters

A classifier can be accurate and still be unreliable. If predictions made at
90% confidence are correct only 70% of the time, downstream systems receive a
misleading account of risk. This repository treats predictive confidence as a
measurable engineering property rather than a cosmetic model output.

## Day 1: establish the measurement layer

The first milestone implements:

- numerically stable softmax;
- multiclass negative log-likelihood;
- multiclass Brier score;
- top-label Expected Calibration Error (ECE); and
- deterministic synthetic logits for a reproducible smoke test.

No empirical model-performance claims are made yet. The synthetic example only
validates the evaluation pipeline and provides a controlled input for later
calibration methods.

## Day 2: fit temperature on held-out logits

The second milestone adds scalar temperature scaling. A bounded optimisation
fits one positive temperature by minimising negative log-likelihood on a
calibration split; evaluation remains strictly held out. Because division by a
positive scalar preserves logit ordering, the calibrated model changes its
confidence without changing its predicted class.

The comparison script reports raw and calibrated NLL, Brier score, ECE, and
accuracy. Its deterministic synthetic output remains a pipeline check rather
than evidence about a trained model.

## Day 3: inspect reliability and estimator sensitivity

The third milestone exposes the bin statistics behind ECE and adds reliability
diagrams that keep empty bins visibly unmeasured. A repeated-seed study reports
mean and sample standard deviation for NLL, Brier score, and ECE at 10, 15, 30,
and 50 bins. This makes an important limitation explicit: histogram-based ECE
is an estimator whose value can change with the binning scheme and sample.

The sensitivity study uses deterministic synthetic data to validate the
pipeline. Its numbers are not claims about a trained vision model.

## Day 4: audit calibration under class imbalance

The fourth milestone partitions top-label calibration by true class and reports
both macro and support-weighted group ECE. It also extends temperature scaling
with optional observation weights, enabling a controlled comparison between a
prevalence-weighted objective and inverse-frequency class balancing.

Group-conditioned ECE is treated as a disparity diagnostic rather than a
replacement for population calibration. The imbalanced synthetic fixture is a
pipeline validation, not a trained-model benchmark.

## Day 5: quantify abstention with risk-coverage curves

The fifth milestone ranks predictions using maximum probability or normalised
negative entropy, then measures the error rate as progressively less-confident
examples are accepted. It reports risk at operational coverage targets and the
discrete area under the risk-coverage curve (AURC).

Tied confidence scores retain input order for deterministic analysis. A deployed
threshold may accept all tied observations, so realised coverage must still be
reported alongside its target.

## Day 6: stress-test clean-data calibration under shift

The sixth milestone introduces controlled logit-space stressors for confidence
softening, stochastic noise, and systematic class bias. A temperature fitted
only on clean calibration data is then evaluated across increasing severities
using accuracy, NLL, Brier score, ECE, AURC, and selective risk.

These transformations isolate different failure mechanisms, but they are not
claimed to reproduce natural image corruptions. Their purpose is to test the
measurement pipeline before running the same protocol on genuine shifted image
data. Severity is defined within each stressor and must not be compared across
stressor families as though it shared a physical unit.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python scripts/evaluate_synthetic.py
python scripts/evaluate_temperature_scaling.py
python scripts/analyze_reliability.py
python scripts/evaluate_class_imbalance.py
python scripts/evaluate_selective_prediction.py
python scripts/evaluate_distribution_shift.py
pytest
```

The scripts print JSON so that future experiment runners can capture output
without parsing human-oriented logs. The reliability analysis also writes a
representative two-panel diagram to `artifacts/reliability_diagram.png`.

## Planned progression

1. Calibration metrics and reproducible synthetic baseline
2. Temperature scaling on held-out logits
3. Reliability diagrams and bin-sensitivity analysis (complete)
4. Calibration under class imbalance (complete)
5. Selective prediction and risk-coverage curves (complete)
6. Distribution-shift stress test (complete)
7. Reproducible report and experiment card

## Metric conventions

- Probabilities are derived from logits with a stable softmax.
- NLL uses the natural logarithm.
- Brier score is the sum of squared class-probability errors, averaged over
  examples.
- ECE uses equal-width confidence bins and top-label correctness.
- Groupwise ECE conditions the same top-label statistic on each true class;
  macro and support-weighted summaries are reported together.
- Selective risk is the error rate among retained predictions; coverage is the
  retained fraction, and AURC is the mean cumulative risk over ranked samples.

These choices are explicit because calibration numbers are only comparable
when their definitions match.
