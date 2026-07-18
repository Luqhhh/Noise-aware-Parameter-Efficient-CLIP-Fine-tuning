"""Test that canonical_image_path maps equivalent paths to the same key."""

import os
import tempfile
from pathlib import Path

import pytest

from common.manifest_loader import canonical_image_path


class TestCanonicalImagePath:
    def test_same_path_returns_same_key(self):
        """Relative path always maps to the same resolved absolute path."""
        with tempfile.TemporaryDirectory() as tmp:
            fname = Path(tmp) / "images" / "abc.jpg"
            fname.parent.mkdir(parents=True)
            fname.touch()
            cwd = os.getcwd()
            try:
                os.chdir(tmp)
                rel = "images/abc.jpg"
                abs_path = str(Path(tmp).resolve() / "images" / "abc.jpg")
                assert canonical_image_path(rel) == canonical_image_path(abs_path)
            finally:
                os.chdir(cwd)

    def test_dot_dot_resolves(self):
        """../ resolves to the same canonical path."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base"
            base.mkdir()
            fname = base / "img.jpg"
            fname.touch()
            key_direct = canonical_image_path(str(fname))
            key_via_dotdot = canonical_image_path(str(base / ".." / "base" / "img.jpg"))
            assert key_direct == key_via_dotdot

    def test_symlink_resolves(self):
        """Symlinked path resolves to same key as real path."""
        with tempfile.TemporaryDirectory() as tmp:
            real_dir = Path(tmp) / "real"
            real_dir.mkdir()
            real_file = real_dir / "img.jpg"
            real_file.touch()
            link_dir = Path(tmp) / "link"
            link_dir.symlink_to(real_dir, target_is_directory=True)
            key_real = canonical_image_path(str(real_file))
            key_link = canonical_image_path(str(link_dir / "img.jpg"))
            assert key_real == key_link

    def test_tilde_expansion(self):
        """~ expands to home directory."""
        home = str(Path.home())
        result = canonical_image_path("~/test.jpg")
        assert result.startswith(home)
        assert "~" not in result

    def test_consistent_trailing_slash_insensitive(self):
        """Key is the string form of the resolved path."""
        with tempfile.TemporaryDirectory() as tmp:
            fname = Path(tmp) / "img.jpg"
            fname.touch()
            key = canonical_image_path(str(fname))
            assert isinstance(key, str)
            assert key == str(fname.resolve())
