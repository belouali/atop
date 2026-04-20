"""
AToP — Attribution-guided Temporal Pattern mining for clinical sequences.

Quick start:

    import atop

    explainer = atop.AToPExplainer.from_bundle("runs/exp01", mimic_dir="/data/mimiciv/hosp")
    explainer.compute_attributions()
    explainer.mine_patterns()
    explainer.validate()
    explainer.report("explanations/")
"""
__version__ = "0.1.0"

from atop.explainer import AToPExplainer
from atop.config import AToPConfig

__all__ = ["AToPExplainer", "AToPConfig"]
