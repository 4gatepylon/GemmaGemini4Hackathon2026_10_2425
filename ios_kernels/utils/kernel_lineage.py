"""Print kernel lineage: one tree per model, from the `parent:` line in each kernel's README.md
header (schema in ios_kernels/CLAUDE.md). Children are ordered by timestamp.

    uv run utils/kernel_lineage.py
"""

import re
import sys
from pathlib import Path

from surfaces.paths import IOS_KERNELS

PARENT_RE = re.compile(r"^parent:\s*(\S+)", re.MULTILINE)


def timestamp(name: str) -> str:
    return name.rsplit("_", 1)[-1]


def read_parent(kernel: Path) -> str | None:
    readme = kernel / "README.md"
    m = PARENT_RE.search(readme.read_text()) if readme.exists() else None
    if m is None:
        print(f"warning: no 'parent:' header in {readme}", file=sys.stderr)
        return None
    return None if m.group(1) == "none" else m.group(1)


def model_tree(model_dir: Path) -> list[str]:
    kernels = sorted((d.name for d in model_dir.glob("kernel_*") if d.is_dir()), key=timestamp)
    children: dict[str | None, list[str]] = {}
    for name in kernels:
        parent = read_parent(model_dir / name)
        if parent is not None and parent not in kernels:
            print(f"warning: {name}: parent {parent} not found in {model_dir.name}/", file=sys.stderr)
            parent = None
        children.setdefault(parent, []).append(name)

    lines = [model_dir.name]

    def walk(parent: str | None, prefix: str) -> None:
        kids = children.get(parent, [])
        for i, name in enumerate(kids):
            last = i == len(kids) - 1
            lines.append(f"{prefix}{'└── ' if last else '├── '}{name}")
            walk(name, prefix + ("    " if last else "│   "))

    walk(None, "")
    return lines


def lineage(kernels_root: Path) -> str:
    models = sorted(d for d in kernels_root.iterdir() if d.is_dir())
    return "\n\n".join("\n".join(model_tree(m)) for m in models)


def main() -> None:
    print(lineage(IOS_KERNELS / "kernels"))


if __name__ == "__main__":
    main()
