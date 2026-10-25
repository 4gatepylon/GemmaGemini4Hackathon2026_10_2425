# CLAUDE.md: ios_kernels

Rules for working in `ios_kernels/`. Design and rationale: `PLAN.md`. Usage: `README.md`.

## If this file is wrong

If anything here is wrong, outdated, or contradicts the code (paths, names, commands, rules),
**stop immediately and tell the user** what is wrong. Don't work around it or silently fix it;
fix this file together with the user first.

## Worktree setup

Large artifacts live in `<checkout>/.context/` (`DATA_DIR` in `surfaces/paths.py`), which is gitignored:

- `.context/models/`: converted weights for every engine (what goes in it: `benchmarking/setup.py`).
- `.context/deps/`: the llama.cpp checkout and build (the benchmark baseline).

The main checkout (the folder that contains the repo's `.git` directory) holds the only copy, created
there by `make -C ios_kernels setup`. Every other worktree (e.g. a Conductor workspace) symlinks those
two folders instead of re-creating them. Link the two folders, not `.context` itself: the rest of a
worktree's `.context/` (e.g. Conductor's notes) stays per-worktree.

From the root of a worktree, before running anything in `ios_kernels/`:

```sh
MAIN="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")"   # the main checkout
mkdir -p .context
ln -s "$MAIN/.context/models" .context/models
ln -s "$MAIN/.context/deps" .context/deps
```

Then check:

- `git status --short` shows nothing under `.context` (it is gitignored); if it shows up, stop and
  tell the user instead of committing it.
- `make -C ios_kernels smoke` starts a server on the shared weights and answers a question.

Notes:

- If `$MAIN/.context/models` doesn't exist, run `make -C ios_kernels setup` in the main checkout first.
- Writes through the links land in the shared copy, which every worktree sees.
- To remove the links: `rm .context/models .context/deps`, **without** a trailing slash. With a
  trailing slash, `rm -r` follows the link and deletes the shared data.
- Without symlinks, launcher commands also accept `--data-dir "$MAIN/.context"`.

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
parent: none | kernel_<name>_<YYYYMMDD-HHMM>   # older folder in the same model folder it was copied from
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
  (`$DATA_DIR`, shared across worktrees: see Worktree setup); build output goes in each kernel's
  gitignored `.build/`.
- **Find kernels by path through `surfaces/paths.py`** (`resolve_kernel`, `KERNELS_DIR`), not by
  hard-coding folder names.
- **Every folder has a `README.md`** whose top is a one-sentence description of the folder.
  "Top" means right after any required top-of-file formatting (the `# <title>` line, and in kernel
  folders the header block above), not before it. Gitignored folders (`.build/`, etc.) are exempt.
- **Docs are written in present tense and built not to go stale quickly**: describe what is
  and point to the source of truth (code, `kernel.json`, scripts) instead of copying volatile
  details like numbers, file lists, or status. Anything only in the scope of one conversation
  (what was just tried, fixed, or discussed) does not go in docs. Exceptions: `PLAN.md`,
  `DESIGN_AND_LIMITATIONS.md`, and `KNOWN_ISSUES.md` files are working notes for agents in the loop
  and can hold plans, status, and history. Because they go stale, each one has, right under its
  `# <title>` line, the `main` commit and the time (to the minute) its content was last checked
  against: `> As of: main @ <short hash>, <YYYY-MM-DD HH:MM TZ>`. Update that line whenever you
  edit the file (the commit your edit is based on, and the current time).
- **Scripts are Python, runnable with `uv run`** (deps in `pyproject.toml`), never bash.
- **Keep changes minimal** and match the surrounding style.

## Utilities (`utils/`)

Small standalone helper scripts, run from `ios_kernels/` with `uv run utils/<script>.py`.
New utility scripts go here.

- `kernel_lineage.py`: kernel lineage, one tree per model (from each README's `parent:` line).
- `check_*.py`: the CI checks (see CI below); run them locally before pushing.

## Checks

From `ios_kernels/`:
- `uv run ruff check .` and `uv run ruff format --check .`: lint and format (config in
  `pyproject.toml`). Both must be clean; `uv run ruff format .` fixes formatting.
- `uv run pytest`: unit tests always run. Kernel e2e tests need `.context/models/*.bin`
  (run `uv run --extra bench benchmarking/setup.py`) and compare exact greedy output against HF.
- `uv run --extra bench python benchmarking/reference.py --kernel <name>`: logits/tokens vs HF.
- `uv run --extra bench benchmarking/bench.py --kernel <name>`: pp512 / tg128 vs llama.cpp and MLX.

## CI (GitHub Actions)

`.github/workflows/ci.yml` runs on every PR and push to `main`. **Every job must pass to merge.**
**Never modify, disable, or work around the workflow or the `utils/check_*.py` scripts** (and never
`git push --no-verify` around them); if a check looks wrong, stop and tell the user.

| job | what must hold | run locally (from `ios_kernels/`) |
|---|---|---|
| ruff | lint and format are clean | `uv run ruff check . && uv run ruff format --check .` |
| kernel tree | `kernels/` matches "Adding a kernel" and the header schema above | `uv run utils/check_kernel_tree.py` |
| no shell scripts | no shell scripts anywhere in the repo (by extension or shebang) | `uv run utils/check_no_shell_scripts.py` |
| forbidden markers | no do-not-submit marker or its three-letter abbreviation (case-insensitive substring, exact strings in `utils/check_forbidden_markers.py`) in lines added or commit messages since `main`, or in the PR title/body | `uv run utils/check_forbidden_markers.py` |
| pytest | `uv run pytest` passes (macOS runner, no weights, so kernel e2e tests skip) | `uv run pytest` |

## PRs

- Base every PR on `main`. Don't stack PRs: a squash-merged base leaves the PR on top
  merged into a dead branch.
