import json
import zipfile

import pytest

from aegis_clip.submission import create_submission, validate_predictions


def test_submission_is_validated_before_publication(tmp_path) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "submission"
    predictions = [("a.jpg", "0001"), ("b.jpg", "0002")]
    manifest = create_submission(
        predictions,
        ["a.jpg", "b.jpg"],
        output,
        checkpoint,
        inference_mode="none",
        tta_risk_acknowledged=False,
        extra_manifest={"corrupt_images": 0},
    )
    assert manifest["prediction_count"] == 2
    assert (output / "pred_results.csv").read_text().splitlines()[0] == "a.jpg,0001"
    persisted = json.loads((output / "manifest.json").read_text())
    assert persisted["corrupt_images"] == 0
    with zipfile.ZipFile(output / "submission.zip") as archive:
        assert archive.namelist() == ["pred_results.csv"]
        assert archive.read("pred_results.csv") == (output / "pred_results.csv").read_bytes()


def test_bad_coverage_leaves_no_csv_or_zip(tmp_path) -> None:
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "submission"
    with pytest.raises(ValueError, match="does not match"):
        create_submission(
            [("a.jpg", "0001")],
            ["a.jpg", "b.jpg"],
            output,
            checkpoint,
            inference_mode="none",
            tta_risk_acknowledged=False,
        )
    assert not (output / "pred_results.csv").exists()
    assert not (output / "submission.zip").exists()


def test_out_of_range_label_is_rejected() -> None:
    with pytest.raises(ValueError, match="Out-of-range"):
        validate_predictions(
            [("a.jpg", "0500")],
            ["a.jpg"],
            valid_labels={f"{index:04d}" for index in range(500)},
        )


def test_nested_image_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="Invalid image name"):
        validate_predictions([("../a.jpg", "0001")], ["../a.jpg"])
