# CLAUDE.md: ios_kernels

Rules for working in `ios_kernels/`. Design and rationale: `PLAN.md`. Usage: `README.md`.

## If this file is wrong

If anything here is wrong, outdated, or contradicts the code (paths, names, commands, rules),
**stop immediately and tell the user** what is wrong. Don't work around it or silently fix it;
fix this file together with the user first.

## Adding a kernel

1. Pick the model folder: `kernels/<model>/`, where `<model>` = HF repo id with `/` -> `__`
   (e.g. `unsloth/Llama-3.2-1B-Instruct` -> `unsloth__Llama-3.2-1B-Instruct`).
   Only `unsloth__Llama-3.2-1B-Instruct` is wired up (hard-coded in `surfaces/paths.py`).
2. Copy the parent kernel folder (usually the newest) to
   `kernels/<model>/kernel_<name>_<YYYYMMDD-HHMM>/`, or start an empty folder for a root kernel.
   - `<name>`: short, lowercase, `_`-separated, describes what the kernel *is*
     (backend, precision, key technique), e.g. `metal_f16`, `metal_q8`, `metal_f16_fusednorm`.
     No model name (it's in the parent folder).
     A derived kernel can reuse its parent's name plus a short tag for the change; keep it short.
   - `<YYYYMMDD-HHMM>`: creation time. The newest timestamp is the default everywhere.
3. Update the header in the new folder's `README.md` (schema below), then the rest of that README.
4. Keep the folder contract (see `surfaces/DESIGN_AND_LIMITATIONS.md#how-to-swap-things`):
   `kernel.json`, build script, weight converter, and a binary with `--serve` + bench flags.
5. Verify (see Checks), then add a row to the table in `kernels/README.md`.
   `uv run utils/kernel_lineage.py` should show the new kernel under its parent.

Layout is always exactly depth 2: `kernels/<model>/kernel_*/`. Don't nest kernels inside
kernels; lineage lives in the header.

## Kernel header schema

The first thing in every kernel's `README.md`, after the `# <folder name>` title:

````
```yaml
kernel: kernel_<name>_<YYYYMMDD-HHMM>   # this folder's name
model: <model folder name>
origin: root | derived                # root = written from scratch
parent: none | kernel_<name>_<YYYYMMDD-HHMM>   # folder this was copied from (same model, unless noted)
change: <one line: what is new vs parent (root: what it is)>
```
````

`uv run utils/kernel_lineage.py` prints the lineage as one tree per model. One `key: value`
per line, so it's also grep-able:

```sh
grep -rh '^parent: ' kernels/*/*/README.md        # all lineage edges
grep -rl '^origin: root' kernels/*/*/README.md    # all root kernels
grep -rl '^parent: kernel_metal_f16_20261024-1620$' kernels/*/*/README.md   # children of one kernel
```

## Rules

- **Kernel folders are immutable** once merged. Iterate by copying. Only doc fixes are allowed in old folders.
- **Large artifacts never go in git**: weights, GGUFs, and llama.cpp go in `<repo>/.context/`
  (`$DATA_DIR`); build output goes in each kernel's gitignored `.build/`.
- **Find kernels by path through `surfaces/paths.py`** (`resolve_kernel`, `KERNELS_DIR`), not by
  hard-coding folder names.
- **Every folder has a `README.md`** whose top is a one-sentence description of the folder.
  "Top" means right after any required top-of-file formatting (the `# <title>` line, and in kernel
  folders the header block above), not before it. Gitignored folders (`.build/`, etc.) are exempt.
- **Docs are written in present tense and built not to go stale quickly**: describe what is
  and point to the source of truth (code, `kernel.json`, scripts) instead of copying volatile
  details like numbers, file lists, or status. Anything only in the scope of one conversation
  (what was just tried, fixed, or discussed) does not go in docs. Exceptions: `PLAN.md` and
  `DESIGN_AND_LIMITATIONS.md` files are working notes for agents in the loop and can hold plans,
  status, and history.
- **Scripts are Python, runnable with `uv run`** (deps in `pyproject.toml`), never bash.
- **Keep changes minimal** and match the surrounding style.

## Utilities (`utils/`)

Small standalone helper scripts, run from `ios_kernels/` with `uv run utils/<script>.py`.
New utility scripts go here.

- `kernel_lineage.py`: kernel lineage, one tree per model (from each README's `parent:` line).

## Checks

From `ios_kernels/`:
- `uv run ruff check .`: lint (config in `pyproject.toml`). Must be clean. `ruff format` is not used.
- `uv run pytest`: unit tests always run. Kernel e2e tests need `.context/models/*.bin`
  (run `uv run --extra bench benchmarking/setup.py`) and compare exact greedy output against HF.
- `uv run --extra bench python benchmarking/reference.py --kernel <name>`: logits/tokens vs HF.
- `uv run --extra bench benchmarking/bench.py --kernel <name>`: pp512 / tg128 vs llama.cpp and MLX.

## PRs

- Base every PR on `main`. Don't stack PRs: a squash-merged base leaves the PR on top
  merged into a dead branch.
