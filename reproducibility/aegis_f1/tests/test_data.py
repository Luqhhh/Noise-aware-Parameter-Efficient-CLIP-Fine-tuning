import json

import pytest

from aegis_clip.data import load_class_mapping


def test_class_mapping_must_be_four_digit_and_contiguous(tmp_path) -> None:
    malformed = tmp_path / "malformed.json"
    malformed.write_text(json.dumps({"cat": 0, "0001": 1}))
    with pytest.raises(ValueError, match="four digits"):
        load_class_mapping(malformed)

    noncontiguous = tmp_path / "noncontiguous.json"
    noncontiguous.write_text(json.dumps({"0000": 0, "0001": 2}))
    with pytest.raises(ValueError, match="contiguous"):
        load_class_mapping(noncontiguous)


def test_valid_competition_mapping_round_trips(tmp_path) -> None:
    path = tmp_path / "classes.json"
    path.write_text(json.dumps({"0000": 0, "0001": 1}))
    class_to_idx, idx_to_class = load_class_mapping(path)
    assert class_to_idx["0001"] == 1
    assert idx_to_class[0] == "0000"
