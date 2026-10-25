"""Fail if the repo contains shell scripts (by extension or shebang); scripts must be Python.
Looks at git-visible files in the whole repo (tracked + untracked-not-ignored).

    uv run utils/check_no_shell_scripts.py
"""

import re
import subprocess
import sys
from pathlib import Path

SHELL_EXTENSIONS = {".sh", ".bash", ".zsh", ".ksh", ".fish"}
SHELL_SHEBANG_RE = re.compile(rb"^#!.*\b(?:sh|bash|zsh|ksh|dash|fish)\b")


def is_shell_script(path: Path) -> bool:
    if path.suffix in SHELL_EXTENSIONS:
        return True
    with path.open("rb") as f:
        return bool(SHELL_SHEBANG_RE.match(f.readline()))


def main() -> None:
    root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    files = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=root, text=True
    ).splitlines()
    bad = [f for f in files if (root / f).is_file() and is_shell_script(root / f)]
    for f in bad:
        print(f"error: {f} is a shell script; rewrite it in Python (run with `uv run`)", file=sys.stderr)
    if bad:
        sys.exit(1)
    print("no shell scripts")


if __name__ == "__main__":
    main()
