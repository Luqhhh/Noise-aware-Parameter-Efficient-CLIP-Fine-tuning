"""The long-side rescale used to simulate the test set's input geometry."""

from __future__ import annotations

from aegis_clip.cli.simulate_input_geometry import target_size


def test_shrinks_the_long_side_and_keeps_the_aspect_ratio():
    assert target_size(1250, 1000, 500) == (500, 400)
    assert target_size(1000, 1250, 500) == (400, 500)


def test_leaves_an_image_already_at_the_target_side_untouched():
    assert target_size(375, 500, 500) == (375, 500)
    assert target_size(500, 375, 500) == (500, 375)


def test_enlarges_when_the_target_side_is_larger():
    assert target_size(200, 100, 400) == (400, 200)


def test_a_square_image_keeps_both_sides_equal():
    assert target_size(1000, 1000, 500) == (500, 500)


def test_the_longer_side_is_exactly_the_requested_one():
    for width, height in ((1300, 956), (866, 1390), (640, 480), (1, 1000)):
        for long_side in (400, 500, 800):
            resized = target_size(width, height, long_side)
            assert max(resized) == long_side, (width, height, long_side, resized)


def test_never_produces_a_zero_dimension():
    # round() is banker's rounding, so a very thin image can round to 0.
    assert target_size(1, 1000, 500) == (1, 500)
    assert target_size(1000, 1, 500) == (500, 1)
    assert min(target_size(1, 4000, 400)) >= 1
