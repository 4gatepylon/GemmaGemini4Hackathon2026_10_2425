"""surfaces/paths.py: kernel lookup across model folders and per-model baseline files."""

import pytest

from surfaces import paths

KERNELS = [
    ("org__A", "kernel_x_20260101-0000"),
    ("org__A", "kernel_y_20260102-0000"),
    ("org__B", "kernel_y_20260103-0000"),
]


@pytest.fixture
def root(tmp_path, monkeypatch):
    for model, kernel in KERNELS:
        (tmp_path / model / kernel).mkdir(parents=True)
        (tmp_path / model / kernel / "kernel.json").write_text("{}")
    monkeypatch.setattr(paths, "KERNELS_ROOT", tmp_path)
    monkeypatch.setattr(paths, "KERNELS_DIR", tmp_path / "org__A")
    monkeypatch.setattr(paths, "MODEL", "org__A")
    monkeypatch.setattr(paths.list_kernels, "__defaults__", ("org__A",))
    monkeypatch.setattr(paths, "BASELINES", {"org__A": {"gguf": "a.gguf"}, "org__B": {"gguf": "b.gguf"}})
    return tmp_path


def test_default_is_newest_kernel_of_default_model(root):
    assert paths.resolve_kernel() == root / "org__A" / "kernel_y_20260102-0000"


def test_substring_searches_default_model_first(root):
    assert paths.resolve_kernel("kernel_y") == root / "org__A" / "kernel_y_20260102-0000"


def test_substring_falls_back_to_other_models(root):
    assert paths.resolve_kernel("20260103") == root / "org__B" / "kernel_y_20260103-0000"


def test_ambiguous_substring_raises(root):
    with pytest.raises(ValueError, match="matched"):
        paths.resolve_kernel("kernel_")


def test_baseline_follows_the_kernels_model(root):
    assert paths.baseline("gguf") == paths.MODELS_DIR / "a.gguf"
    assert paths.baseline("gguf", paths.resolve_kernel("20260103")) == paths.MODELS_DIR / "b.gguf"
