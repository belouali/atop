"""
AToP, Attribution-guided Temporal Pattern mining for clinical sequences.

Quick start:

    import atop

    explainer = atop.AToPExplainer.from_bundle("runs/exp01", mimic_dir="/data/mimiciv/hosp")
    explainer.compute_attributions()
    explainer.mine_patterns()
    explainer.validate()
    explainer.report("explanations/")

Note: ``AToPExplainer`` and ``AToPConfig`` are exposed via PEP-562 lazy
imports so that scripts using only torch-free submodules (e.g.
``from atop.plots import dumbbell``, ``from atop.run_layout import RunLayout``)
do not transitively pull torch. The lazy attributes resolve on first access.
"""
__version__ = "0.1.0"

__all__ = ["AToPExplainer", "AToPConfig"]


def __getattr__(name):
    # PEP 562, defer the heavy explainer import until someone actually asks.
    if name == "AToPExplainer":
        from atop.explainer import AToPExplainer  # noqa: F401
        return AToPExplainer
    if name == "AToPConfig":
        from atop.config import AToPConfig  # noqa: F401
        return AToPConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
