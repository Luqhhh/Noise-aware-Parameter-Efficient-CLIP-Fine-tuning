import pytest
import torch

from aegis_clip.prompt_audit import numeric_prompt_diagnostics


def test_numeric_prompt_diagnostics_reports_geometry_and_accuracy() -> None:
    text = torch.eye(3)
    images = torch.eye(3)
    report = numeric_prompt_diagnostics(
        text_features=text,
        image_features=images,
        labels=torch.tensor([0, 1, 2]),
        clean_probability=torch.ones(3),
        classifier_weights=text,
    )
    assert report["raw_accuracy"] == 1.0
    assert report["clean_core_accuracy"] == 1.0
    assert report["unique_predicted_classes"] == 3
    assert report["text_pairwise_off_diagonal"]["mean"] == 0.0
    assert report["same_id_alignment_with_classifier"]["mean"] == 1.0


def test_numeric_prompt_diagnostics_fails_on_empty_clean_core() -> None:
    with pytest.raises(ValueError, match="clean core is empty"):
        numeric_prompt_diagnostics(
            text_features=torch.eye(2),
            image_features=torch.eye(2),
            labels=torch.tensor([0, 1]),
            clean_probability=torch.zeros(2),
            classifier_weights=torch.eye(2),
        )
