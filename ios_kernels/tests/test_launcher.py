"""ios_kernels_launcher.py + Makefile: the Makefile stays a pure shorthand for the launcher."""

import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

import ios_kernels_launcher as launcher

HERE = Path(launcher.__file__).parent
needs_make = pytest.mark.skipif(shutil.which("make") is None, reason="needs make")


def makefile_commands() -> set[str]:
    text = (HERE / "Makefile").read_text().replace("\\\n", " ")
    return set(re.search(r"^COMMANDS := (.+)$", text, re.M).group(1).split())


def test_makefile_and_launcher_have_the_same_commands():
    assert makefile_commands() == set(launcher.COMMANDS)


@needs_make
def test_make_help_lists_every_command():
    out = subprocess.run(["make", "-s", "-C", HERE, "help"], capture_output=True, text=True, check=True).stdout
    for name in launcher.COMMANDS:
        assert re.search(rf"^\s+{re.escape(name)}\s", out, re.M), name
    assert "make help args=serve" in out


@needs_make
def test_make_help_for_one_command():
    out = subprocess.run(
        ["make", "-s", "-C", HERE, "help", "args=bench-ours"], capture_output=True, text=True, check=True
    ).stdout
    assert "--define NAME=VALUE" in out and "--kernel KERNEL" in out


@needs_make
def test_make_forwards_args_verbatim():
    out = subprocess.run(
        ["make", "-n", "-C", HERE, "serve", "args=--port 9 --kernel x"], capture_output=True, text=True, check=True
    ).stdout
    assert "ios_kernels_launcher.py serve --port 9 --kernel x" in out


def test_unknown_flags_rejected_unless_passthrough():
    with pytest.raises(SystemExit) as e:
        launcher.main(["kernels", "--bogus"])
    assert e.value.code == 2
    args, extra = launcher.build_parser().parse_known_args(["serve", "--port", "9", "--fake-delay", "0"])
    assert (args.port, extra) == (9, ["--fake-delay", "0"])


def test_kernels_marks_newest_as_default(capsys):
    launcher.main(["kernels"])
    newest = launcher.paths().list_kernels()[-1].name
    assert f"* {newest}" in capsys.readouterr().out


def test_new_kernel_copies_without_build_output(tmp_path):
    model_dir = tmp_path / "org__Model"
    src = model_dir / "kernel_a_20260101-0000"
    (src / ".build").mkdir(parents=True)
    (src / ".build" / "llm").write_text("binary")
    (src / "kernel.json").write_text("{}")
    (src / "README.md").write_text(
        "# kernel_a_20260101-0000\n\n```yaml\nkernel: kernel_a_20260101-0000\n"
        "model: org__Model\norigin: root\nparent: none\nchange: first\n```\n\nBody.\n"
    )
    dst = launcher.new_kernel("metal_q8", src, model_dir, now=datetime(2026, 10, 24, 17, 5))
    assert dst.name == "kernel_metal_q8_20261024-1705"
    assert (dst / "kernel.json").exists() and not (dst / ".build").exists()
    # Lineage header (CLAUDE.md schema) now describes the copy, and the body is kept.
    assert (dst / "README.md").read_text() == (
        "# kernel_metal_q8_20261024-1705\n\n```yaml\nkernel: kernel_metal_q8_20261024-1705\nmodel: org__Model\n"
        "origin: derived\nparent: kernel_a_20260101-0000\nchange: TODO: one line, what is new vs parent\n```\n"
        "\nBody.\n"
    )
    with pytest.raises(SystemExit, match="already exists"):
        launcher.new_kernel("metal_q8", src, model_dir, now=datetime(2026, 10, 24, 17, 5))
    with pytest.raises(SystemExit, match="bad name"):
        launcher.new_kernel("Bad-Name", src, model_dir)


def test_new_kernel_header_matches_real_kernel_readme():
    # The header regex must match the format the real kernels use (CLAUDE.md schema).
    text = (launcher.paths().list_kernels()[-1] / "README.md").read_text()
    assert re.match(r"# [^\n]*\n+```yaml\n.*?\n```\n", text, re.S)


def test_data_dir_flag_reaches_tools(tmp_path):
    # In a fresh process (as from make), --data-dir must win over the default .context/.
    r = subprocess.run(
        [sys.executable, HERE / "ios_kernels_launcher.py", "serve-llamacpp", "--data-dir", tmp_path],
        capture_output=True,
        text=True,
        cwd=HERE,
    )
    assert r.returncode != 0 and str(tmp_path / "models" / "llama1b_f16.gguf") in r.stderr


def test_smoke_with_fake_runner(capfd):
    assert launcher.main(["smoke", "--runner", "fake", "--prompt", "ping pong"]) == 0
    assert "Echo: ping pong" in capfd.readouterr().out
