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

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python scripts/evaluate_synthetic.py
pytest
```

The evaluation script prints a JSON object so that future experiment runners
can capture the output without parsing human-oriented logs.

## Planned progression

1. Calibration metrics and reproducible synthetic baseline
2. Temperature scaling on held-out logits
3. Reliability diagrams and bin-sensitivity analysis
4. Calibration under class imbalance
5. Selective prediction and risk-coverage curves
6. Distribution-shift stress test
7. Reproducible report and experiment card

## Metric conventions

- Probabilities are derived from logits with a stable softmax.
- NLL uses the natural logarithm.
- Brier score is the sum of squared class-probability errors, averaged over
  examples.
- ECE uses equal-width confidence bins and top-label correctness.

These choices are explicit because calibration numbers are only comparable
when their definitions match.
