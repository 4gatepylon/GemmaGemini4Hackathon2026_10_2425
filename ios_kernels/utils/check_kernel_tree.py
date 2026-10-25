"""Check kernels/ against the layout and kernel header schema in ios_kernels/CLAUDE.md.
Prints every problem and exits 1 if there are any. Looks at git-visible files only
(tracked + untracked-not-ignored), so build output and other ignored files don't count.

    uv run utils/check_kernel_tree.py
"""

import json
import re
import subprocess
import sys
from pathlib import Path

KERNELS_DIR = Path(__file__).resolve().parents[1] / "kernels"
MODEL_RE = re.compile(r"^[A-Za-z0-9._-]+__[A-Za-z0-9._-]+$")  # HF repo id with "/" -> "__"
KERNEL_RE = re.compile(r"^kernel_[a-z0-9]+(?:_[a-z0-9]+)*_\d{8}-\d{4}$")
HEADER_RE = re.compile(r"\A# (?P<title>.+)\n\n```yaml\n(?P<body>.*?)\n```\n", re.DOTALL)
HEADER_KEYS = ["kernel", "model", "origin", "parent", "change"]
MANIFEST_KEYS = ["hf_repo", "weights", "build", "binary", "convert"]


def git_files(root: Path) -> list[str]:
    """Paths (relative to root) of files git sees under root."""
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "--", "."],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [f for f in out.splitlines() if (root / f).exists()]


def timestamp(kernel: str) -> str:
    return kernel.rsplit("_", 1)[-1]


def check_header(kernel_dir: Path, model: str, siblings: set[str]) -> list[str]:
    where = f"{model}/{kernel_dir.name}/README.md"
    m = HEADER_RE.match((kernel_dir / "README.md").read_text())
    if m is None:
        return [f"{where}: must start with '# {kernel_dir.name}', a blank line, then a ```yaml header block"]
    problems = []
    if m["title"] != kernel_dir.name:
        problems.append(f"{where}: title must be '# {kernel_dir.name}'")
    pairs = [line.split(":", 1) for line in m["body"].splitlines()]
    if any(len(p) != 2 for p in pairs) or [k.strip() for k, _ in pairs] != HEADER_KEYS:
        return [*problems, f"{where}: header must be exactly these 'key: value' lines in order: {HEADER_KEYS}"]
    h = {k.strip(): v.strip() for k, v in pairs}
    if h["kernel"] != kernel_dir.name:
        problems.append(f"{where}: 'kernel: {h['kernel']}' must be the folder name")
    if h["model"] != model:
        problems.append(f"{where}: 'model: {h['model']}' must be the model folder name '{model}'")
    if h["origin"] == "root":
        if h["parent"] != "none":
            problems.append(f"{where}: a root kernel must have 'parent: none'")
    elif h["origin"] == "derived":
        if h["parent"] not in siblings:
            problems.append(f"{where}: 'parent: {h['parent']}' is not a kernel folder in {model}/")
        elif timestamp(h["parent"]) >= timestamp(kernel_dir.name):
            problems.append(f"{where}: parent {h['parent']} must be older than this kernel")
    else:
        problems.append(f"{where}: 'origin' must be 'root' or 'derived'")
    if not h["change"]:
        problems.append(f"{where}: 'change' must not be empty")
    return problems


def check_manifest(kernel_dir: Path, model: str) -> list[str]:
    where = f"{model}/{kernel_dir.name}/kernel.json"
    try:
        manifest = json.loads((kernel_dir / "kernel.json").read_text())
    except json.JSONDecodeError as e:
        return [f"{where}: invalid JSON ({e})"]
    missing = [k for k in MANIFEST_KEYS if k not in manifest]
    if missing:
        return [f"{where}: missing keys {missing}"]
    problems = []
    if manifest["hf_repo"].replace("/", "__") != model:
        problems.append(f"{where}: hf_repo '{manifest['hf_repo']}' does not match the model folder '{model}'")
    for key in ("build", "convert"):
        if not (kernel_dir / manifest[key]).is_file():
            problems.append(f"{where}: {key} file '{manifest[key]}' does not exist")
    return problems


def check(kernels_dir: Path, files: list[str]) -> list[str]:
    """All layout problems for `files` (paths relative to kernels_dir)."""
    problems = []
    if "README.md" not in files:
        problems.append("kernels/README.md is missing")
    tree: dict[str, dict[str, list[str]]] = {}  # model -> kernel -> files inside the kernel folder
    for f in files:
        parts = f.split("/")
        if len(parts) == 1:
            if f != "README.md":
                problems.append(f"kernels/{f}: only README.md and model folders belong in kernels/")
        elif len(parts) == 2:
            tree.setdefault(parts[0], {})
            if parts[1] != "README.md":
                problems.append(f"{f}: only README.md and kernel folders belong in a model folder")
        else:
            tree.setdefault(parts[0], {}).setdefault(parts[1], []).append("/".join(parts[2:]))

    for model, kernels in sorted(tree.items()):
        if not MODEL_RE.match(model):
            problems.append(f"{model}/: model folder must be the HF repo id with '/' -> '__'")
        if not (kernels_dir / model / "README.md").is_file():
            problems.append(f"{model}/README.md is missing")
        for kernel, inner in sorted(kernels.items()):
            kernel_dir = kernels_dir / model / kernel
            if not KERNEL_RE.match(kernel):
                problems.append(
                    f"{model}/{kernel}/: name must be kernel_<name>_<YYYYMMDD-HHMM> (<name> lowercase, '_'-separated)"
                )
                continue
            nested = sorted({d for p in inner for d in p.split("/")[:-1] if KERNEL_RE.match(d)})
            problems += [f"{model}/{kernel}/: contains kernel folder {n}/; kernels must not be nested" for n in nested]
            missing = [f for f in ("README.md", "kernel.json") if f not in inner]
            if missing:
                problems.append(f"{model}/{kernel}/: missing {missing}")
                continue
            problems += check_manifest(kernel_dir, model)
            problems += check_header(kernel_dir, model, set(kernels))
    return problems


def main() -> None:
    problems = check(KERNELS_DIR, git_files(KERNELS_DIR))
    for p in problems:
        print(f"error: {p}", file=sys.stderr)
    if problems:
        sys.exit(1)
    print("kernels/ layout OK")


if __name__ == "__main__":
    main()
