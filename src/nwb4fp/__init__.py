"""nwb4fp — convert Open-Ephys electrophysiology and DeepLabCut behaviour to NWB.

The two entry points that drive the conversion pipeline are :func:`test_qmnwb`
(pre-conversion checks) and :func:`run_qmnwb` (the conversion itself). They are
exposed lazily so that ``import nwb4fp`` stays cheap and does not require the heavy
optional dependencies (SpikeInterface, pynwb, neuroconv) just to read metadata.
"""

__version__ = "0.9.0"

__all__ = ["__version__", "run_qmnwb", "test_qmnwb"]


def __getattr__(name):
    if name in ("run_qmnwb", "test_qmnwb"):
        from nwb4fp.main.main_create_nwb import run_qmnwb, test_qmnwb

        return {"run_qmnwb": run_qmnwb, "test_qmnwb": test_qmnwb}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
