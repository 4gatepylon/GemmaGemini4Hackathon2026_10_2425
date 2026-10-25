# CLAUDE.md: ios_kernels

Rules for working in `ios_kernels/`. Design and rationale: `PLAN.md`. Usage: `README.md`.

## If this file is wrong

If anything here is wrong, outdated, or contradicts the code (paths, names, commands, rules),
**stop immediately and tell the user** what is wrong. Don't work around it or silently fix it;
fix this file together with the user first.

## Large files and worktrees

Every large artifact except HF downloads has exactly one copy, in the **main checkout's**
`.context/` (gitignored; `DATA_DIR` in `surfaces/paths.py`). The main checkout is the folder that
contains the repo's `.git` directory. Every other worktree (e.g. a Conductor workspace) symlinks
folders from it.

- `models/`: converted weights for every model and every format: our kernels' `.bin` files, GGUFs
  (quantized ones included), MLX conversions.
- `deps/`: the llama.cpp checkout and build (the benchmark baseline).
- Any other large output worth keeping (e.g. profiling traces) gets its own folder, e.g. `prof/`.
  Create it in the main checkout first, then link it like the others.

HF downloads live in the global HF cache (`~/.cache/huggingface/`), which every worktree already
shares.

**Never leave a large artifact only in a worktree's own `.context/`.** When Conductor archives a
workspace, it moves that workspace's `.context/` to `~/conductor/archived-contexts/`, where
nothing finds it.

Commands below use `$MAIN` for the main checkout. Each agent shell call starts fresh, so set it in
the same call with the first line of the snippet below. In commands you hand to the user, write
out the absolute path.

### Setting up a worktree

From the root of the worktree, before running anything in `ios_kernels/`:

```sh
MAIN="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")"   # the main checkout
mkdir -p .context
[ -e .context/models ] || ln -s "$MAIN/.context/models" .context/models
[ -e .context/deps ]   || ln -s "$MAIN/.context/deps"   .context/deps
```

The guards matter: `ln -s` onto an existing folder or link silently creates a link *inside* it
(e.g. `$MAIN/.context/models/models`). Link whole folders, never single files and never `.context`
itself; the rest of a worktree's `.context/` (Conductor's notes, small scratch output) stays
per-worktree. To remove a link: `rm .context/models`, **without** a trailing slash (with one,
`rm -r` follows the link and deletes the shared data).

Then check:

- `ls -ld .context/models .context/deps` shows both as links (`->`) into `$MAIN/.context/`. If one
  is a real folder, stop and tell the user: its contents have to end up in `$MAIN/.context/`
  (llama.cpp gets rebuilt there, not moved) before the folder is replaced by the link.
- `git status --short` shows nothing under `.context`. If it does, stop and tell the user instead
  of committing it.
- `ls -L .context/models/` has your kernel's `weights` file (named in its `kernel.json`) and its
  model's baseline files (named in `BASELINES` in `surfaces/paths.py`). If one is missing, see
  "Missing artifacts" below.
- `.context/deps/llama.cpp/build/bin/llama-bench --help` runs. A `Library not loaded: @rpath/...`
  error means the path the build was made at no longer exists: the build was moved, or it was
  built through a worktree's link and that worktree is gone. See "Rebuilding llama.cpp" below.
- `make -C ios_kernels smoke` starts a server on the shared weights and answers a question.

### Creating artifacts

- Write new artifacts straight into `$MAIN/.context/`, so every worktree sees them at once:
  `--out "$MAIN/.context/models/<file>"` for a kernel's `convert_weights.py` (llama.cpp and mlx-lm
  tools have their own output flags; see `benchmarking/setup.py`), and `--data-dir "$MAIN/.context"`
  for launcher commands that write, e.g. `make -C ios_kernels setup args="--kernel <name> --data-dir $MAIN/.context"`.
- A new format gets a new file name that says what it is (e.g. `..._q8_0.gguf`). Set it as the new
  kernel's `weights` in `kernel.json`; a copied kernel still names its parent's file.
- Only overwrite, rename or delete shared files you created yourself. Other worktrees may be using
  the rest.
- If a write stops partway (error, Ctrl-C, full disk), delete the file it was writing before you
  retry. `setup` skips anything that exists, even a truncated file.

### Rebuilding llama.cpp

**Never move a build tree.** llama.cpp's binaries load their libraries from the absolute path they
were built in. Build it only at its final place, `$MAIN/.context/deps/llama.cpp`, by passing
`--data-dir "$MAIN/.context"` to `setup` (not through a worktree's link).

`setup` skips llama.cpp while `build/bin/llama-bench` exists, even if it is broken, and cmake
refuses to reuse a build folder that was moved. To rebuild, ask the user to run
`rm -rf <main checkout>/.context/deps/llama.cpp/build`, then run the `setup` command above with a
kernel whose weights already exist (otherwise it also downloads and converts them).

### Missing artifacts

1. For a missing plain file or folder (weights, GGUFs, MLX conversions, traces), look for a
   stranded copy first: `find ~/conductor/archived-contexts -name '<file>'`. If you find one, tell
   the user and give them `mv -n <found> <main checkout>/.context/models/` with absolute paths
   rather than moving shared data yourself. Never do this for llama.cpp: rebuild it instead.
2. Otherwise re-create it with the `setup` command above. It writes the kernel's weights and
   baselines, skipping what already exists. Check free disk first (`df -h`); weights are several
   GB each.

## Adding a kernel

1. Pick the model folder: `kernels/<model>/`, where `<model>` = HF repo id with `/` -> `__`
   (e.g. `unsloth/Llama-3.2-1B-Instruct` -> `unsloth__Llama-3.2-1B-Instruct`).
   The default model is `MODEL` in `surfaces/paths.py`; other models' kernels are picked with
   `--kernel`. A new model folder also needs its baseline file names (GGUF, MLX) in `BASELINES`
   there, and its HF reference text in `surfaces/tests/test_kernel_e2e.py` (`SKY_REFS`).
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
- **Large artifacts never go in git**: weights, GGUFs, and llama.cpp go in the main checkout's
  `.context/`, symlinked into worktrees ("Large files and worktrees" above); build output goes in
  each kernel's gitignored `.build/`.
- **Find kernels by path through `surfaces/paths.py`** (`resolve_kernel`, `KERNELS_ROOT`, `baseline`), not by
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
- `uv run pytest`: unit tests always run. Kernel e2e tests need the default kernel's weights in
  `.context/models/` (see "Missing artifacts" above) and compare exact greedy output against HF.
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
