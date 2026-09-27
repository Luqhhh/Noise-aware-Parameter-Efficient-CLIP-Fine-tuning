"""Opt-in P75 runtime hook; placed first on PYTHONPATH only for patch runs."""
from p75_patch_readout import patch_runtime

patch_runtime()
