"""utils/check_*.py: the CI checks accept the real repo and catch the problems they exist for."""

import json
from pathlib import Path

import pytest

from utils import check_forbidden_markers, check_kernel_tree, check_no_shell_scripts

MODEL = "org__Model-1B"
ROOT = "kernel_metal_f16_20260101-0000"
CHILD = "kernel_metal_f16_fused_20260102-0000"


def header(kernel: str, origin: str = "root", parent: str = "none", model: str = MODEL) -> str:
    return (
        f"# {kernel}\n\n```yaml\nkernel: {kernel}\nmodel: {model}\norigin: {origin}\n"
        f"parent: {parent}\nchange: something\n```\n\nOne-sentence description.\n"
    )


def make_kernel(kernels: Path, kernel: str, readme: str, hf_repo: str = "org/Model-1B") -> None:
    d = kernels / MODEL / kernel
    d.mkdir(parents=True)
    (d / "README.md").write_text(readme)
    (d / "build.py").write_text("")
    (d / "convert_weights.py").write_text("")
    manifest = {
        "hf_repo": hf_repo,
        "weights": "w.bin",
        "build": "build.py",
        "binary": ".build/llm",
        "convert": "convert_weights.py",
    }
    (d / "kernel.json").write_text(json.dumps(manifest))


def problems(kernels: Path) -> list[str]:
    files = [str(p.relative_to(kernels)) for p in kernels.rglob("*") if p.is_file()]
    return check_kernel_tree.check(kernels, files)


@pytest.fixture
def kernels(tmp_path: Path) -> Path:
    (tmp_path / MODEL).mkdir()
    (tmp_path / "README.md").write_text("kernels\n")
    (tmp_path / MODEL / "README.md").write_text("model\n")
    make_kernel(tmp_path, ROOT, header(ROOT))
    make_kernel(tmp_path, CHILD, header(CHILD, "derived", ROOT))
    return tmp_path


def test_real_kernel_tree_is_valid():
    kernels = check_kernel_tree.KERNELS_DIR
    assert check_kernel_tree.check(kernels, check_kernel_tree.git_files(kernels)) == []


def test_valid_tree(kernels):
    assert problems(kernels) == []


def write(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


BAD_MANIFEST = json.dumps({"hf_repo": "x/y", "weights": "", "build": "", "binary": "", "convert": ""})
NESTED = "kernel_x_20260103-0000"
BROKEN = [  # (how to break the valid tree, expected problem substring)
    (lambda k: write(k / "stray.txt"), "only README.md and model folders"),
    (lambda k: write(k / MODEL / "notes.md"), "only README.md and kernel folders"),
    (lambda k: (k / MODEL / "README.md").unlink(), "README.md is missing"),
    (lambda k: write(k / MODEL / "kernel_Bad-Name_2026" / "x"), "name must be kernel_<name>_<YYYYMMDD-HHMM>"),
    (lambda k: make_kernel(k / MODEL / ROOT, NESTED, header(NESTED)), "must not be nested"),
    (lambda k: write(k / MODEL / CHILD / "README.md", f"# {CHILD}\n\nno header\n"), "yaml header block"),
    (lambda k: write(k / MODEL / CHILD / "README.md", header(ROOT, "derived", ROOT)), "must be the folder name"),
    (lambda k: write(k / MODEL / CHILD / "README.md", header(CHILD, model="other__M")), "model folder name"),
    (
        lambda k: write(k / MODEL / CHILD / "README.md", header(CHILD, "derived", "kernel_x_20250101-0000")),
        "is not a kernel folder",
    ),
    (lambda k: write(k / MODEL / ROOT / "README.md", header(ROOT, "derived", CHILD)), "must be older"),
    (lambda k: write(k / MODEL / ROOT / "README.md", header(ROOT, "root", CHILD)), "parent: none"),
    (lambda k: write(k / MODEL / ROOT / "README.md", header(ROOT, "copied")), "'root' or 'derived'"),
    (lambda k: write(k / MODEL / ROOT / "kernel.json", "{"), "invalid JSON"),
    (lambda k: (k / MODEL / ROOT / "build.py").unlink(), "build file 'build.py' does not exist"),
    (lambda k: write(k / MODEL / ROOT / "kernel.json", BAD_MANIFEST), "does not match the model folder"),
]


@pytest.mark.parametrize("break_it, expected", BROKEN)
def test_broken_tree(kernels, break_it, expected):
    break_it(kernels)
    assert any(expected in p for p in problems(kernels)), problems(kernels)


def test_shell_script_detection(tmp_path: Path):
    files = {
        "a.sh": "echo",
        "b.py": "#!/usr/bin/env python3\n",
        "c": "#!/bin/bash\necho",
        "d": "#!/usr/bin/env zsh\n",
        "e.txt": "sh is a word\n",
    }
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    assert {n for n in files if check_no_shell_scripts.is_shell_script(tmp_path / n)} == {"a.sh", "c", "d"}


def test_find_markers():
    first, second = check_forbidden_markers.MARKERS
    assert check_forbidden_markers.find_markers("all good") == []
    assert check_forbidden_markers.find_markers(f"x {first.lower()} y") == [first]
    assert check_forbidden_markers.find_markers(f"pre{second.title()}post") == [second]
