import copy
import json
from pathlib import Path

import pytest

from scripts.report_rematch750_v3 import compare


BASELINE = json.loads((Path(__file__).resolve().parents[1] / "results/rematch750_v2_npu32.json").read_text())


def candidate(name, macro, seconds):
    row = copy.deepcopy(BASELINE)
    row.update(candidate=name, raw_macro=macro, train_cli_seconds=seconds)
    return row


def test_fastest_full_run_must_meet_macro_floor():
    rows = [
        candidate("RM_V3_B64", BASELINE["raw_macro"] - 0.0005, 1800),
        candidate("RM_V3_B128_E16", BASELINE["raw_macro"] - 0.0009, 1700),
        candidate("RM_V3_B1024_E16_LR4", BASELINE["raw_macro"] - 0.02, 900),
    ]
    report = compare(BASELINE, rows)
    assert report["winner"] == "RM_V3_B128_E16"
    assert report["decision"] == "new_efficiency_candidate"


def test_out_of_scope_conditional_run_is_rejected():
    rows = [
        candidate("RM_V3_B64", BASELINE["raw_macro"], 1800),
        candidate("RM_V3_B128_E16", BASELINE["raw_macro"] - 0.01, 1700),
        candidate("RM_V3_B1024_E16_LR4", BASELINE["raw_macro"] - 0.02, 900),
        candidate("RM_V3_B64_LR2", BASELINE["raw_macro"], 1500),
    ]
    with pytest.raises(ValueError, match="only authorized"):
        compare(BASELINE, rows)
