# AToP: Attribution-guided Temporal Pattern Mining for EHR Explainability

AToP is a post-hoc explainability framework for Transformer-based clinical prediction models trained on longitudinal Electronic Health Record (EHR) data. It transforms per-token Integrated Gradients attributions into structured **temporal pattern narratives** — revealing not just which tokens matter, but which *sequences of clinical events across encounters* drive model predictions.

## Why AToP?

Standard XAI methods (SHAP, IG, attention) produce flat per-token importance scores. These miss the temporal structure that makes clinical reasoning meaningful: a diagnosis in visit 1 followed by a treatment in visit 3 is a clinically different signal than either event in isolation. AToP mines these cross-visit trajectories and validates them against model behavior.

## Framework Overview

AToP operates in four stages on a trained Transformer:

1. **Attribution** — Compute per-token Integrated Gradients; retain top 80% of attribution mass per patient
2. **Mapping** — Map salient tokens back to their original admission-block structure
3. **Mining** — Mine cross-visit temporal patterns via frequent episode mining on training-set salient blocks
4. **Validation** — Evaluate patterns on held-out test data; assess model sensitivity via token masking

## Installation

```bash
pip install git+https://github.com/belouali/atop.git
```

Or clone and install locally:

```bash
git clone https://github.com/belouali/atop.git
cd atop
pip install -e .
```

## Quick Start

```python
import atop

# Load a trained model bundle (produced by scripts/train_model.py)
explainer = atop.AToPExplainer.from_bundle(
    run_dir="runs/exp01",
    mimic_dir="/data/mimiciv/hosp"
)

# Run the full pipeline
explainer.compute_attributions()
explainer.mine_patterns(method="episode")
explainer.validate(top_k=15)
explainer.report(out_dir="explanations/")
```

## Reproducing Paper Results

See `scripts/` for the full pipeline:

```bash
# 1. Preprocess MIMIC-IV and train the Transformer
python scripts/preprocess.py --mimic_dir /data/mimiciv/hosp --out_dir runs/exp01
python scripts/train_model.py --run_dir runs/exp01

# 2. Run AToP explanation pipeline
python scripts/run_atop.py --run_dir runs/exp01 --mimic_dir /data/mimiciv/hosp

# 3. Compare against LACE+ baseline
python scripts/compare_baselines.py --run_dir runs/exp01
```

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- MIMIC-IV v2.2+ (requires credentialed access via PhysioNet)

## Citation

> Belouali A, Kharrazi H. AToP: An Explainability Framework for Transformer-Based EHR Predictions
> Using Attribution-Guided Temporal Pattern Mining. *(under review)*

## License

MIT License. See `LICENSE` for details.

## Acknowledgments

This work uses [MIMIC-IV](https://physionet.org/content/mimiciv/), a freely accessible critical care database.
Data access requires credentialed PhysioNet registration.
