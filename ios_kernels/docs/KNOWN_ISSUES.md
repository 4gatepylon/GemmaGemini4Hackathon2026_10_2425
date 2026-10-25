# Known issues

> As of: main @ 28fd54e, 2026-10-25 12:37 PDT

File/line references point at that commit. Every issue has a repro: re-run it before acting,
because fixes may have landed since.

Paths are relative to `ios_kernels/`. "Launcher" is `ios_kernels_launcher.py`; `make <cmd>` runs it.

## High priority

One small PR each, every one based on `main`.

### 1. Passing `args=` to `make test` / `make test-fast` breaks a launcher test

- **Where:** `tests/test_launcher.py:32` (`test_make_help_lists_every_command`).
- **Repro:** `make test-fast args="tests/test_launcher.py"` → `1 failed` (`make help` exits 2).
- **Cause:** make passes command-line variables to child makes through `$MAKEFLAGS`. The test's
  nested `make -C . help` inherits `args=tests/test_launcher.py` and runs
  `launcher help tests/test_launcher.py` ("unknown command").
- **Proposed PR:** a `make()` helper in the test file that runs make with `MAKEFLAGS`,
  `MAKELEVEL` and `MFLAGS` removed from the environment, used by every make-based test.
  Verified by running the suite through `make test-fast args=...`.

### 2. A second SIGTERM right after the first can still orphan the server

- **Where:** `on_terminate` / `run()` in the launcher (launcher:37-56).
- **Repro:** start the make recipe, `uv run --quiet --project . ios_kernels_launcher.py serve-fake
  --port <p>`, then `pkill -TERM -f "ios_kernels_launcher.py serve-fake --port <p>"`. The server
  keeps answering in about a third of runs (4/12 here).
- **Cause:** the handler raises `Terminated` and `run()` forwards the signal from
  `except Terminated`. `pkill -f` matches both the outer `uv run` and the launcher, so the launcher
  gets two SIGTERMs microseconds apart. The second one raises inside the `except` block before
  `send_signal` runs, so nothing is forwarded. Related:
  - A signal that arrives while `Popen()` is starting the child escapes `run()` (child orphaned).
  - When the direct child dies of the forwarded signal, `run()` returns -15 and the launcher exits
    241 instead of 128 + signum.
  - A forwarded signal doesn't stop multi-step commands (`bench-ours`, `serve-llamacpp`) from
    starting their next step if the first step exits 0.
- **Proposed PR:** while `run()` has a child, have the handler only record the signal and forward
  it (no raising). Send it after `Popen()` returns if it arrived during start-up, and raise
  `Terminated` once the child has exited. A unit test can't reliably hit the microsecond window,
  so verify with a scripted `pkill -f` loop and keep the existing single-signal test.

### 3. `make ask` fails when `--max-tokens` / `--system` come before the question

- **Where:** `client_flags` (launcher:308) and the `ask` registration; README's command table
  lists these flags for `chat` and `ask`.
- **Repro:** `uv run ios_kernels_launcher.py ask --max-tokens 3 why is the sky blue` →
  `error: argument --max-tokens: expected one argument`. It works only with the flags after
  the question.
- **Cause:** the flags aren't declared on the launcher's `ask` parser; they are unknown flags
  passed through, the `nargs="+"` prompt positional takes their values, and `chat.py` then
  gets `--max-tokens` with no value.
- **Proposed PR:** declare `--max-tokens`, `--system`, `--model` in `client_flags` and forward
  them to `chat.main` (they then also show up in `make help args=ask`); a parse test with the
  flags before the prompt.

### 4. `make smoke` hides the server's errors

- **Where:** `cmd_smoke`, launcher:166 (`stdout`/`stderr` = `DEVNULL`).
- **Repro:** `uv run ios_kernels_launcher.py smoke --data-dir /tmp/empty` prints only
  `server exited early (code 1)`; the real cause (the weights file is missing) is discarded.
  A slow start (e.g. a first-time kernel build) shows nothing for up to 120 s.
- **Proposed PR:** write the server's output to a temp file and print its tail when the server
  exits early or never gets healthy; a test that `smoke --data-dir <empty dir>` shows the
  "missing" message.

### 5. Tracebacks instead of one-line errors

- **Where:** `kernel_dir` (launcher:59); `cmd_chat` / `cmd_ask` (launcher:145, 153).
- **Repro:** `uv run ios_kernels_launcher.py build --kernel nope` → `ValueError` traceback.
  `uv run ios_kernels_launcher.py ask --port 1 hi` → ~100 lines ending in
  `openai.APIConnectionError: Connection error.`, without the URL or a hint.
- **Proposed PR:** turn `ValueError`/`FileNotFoundError` from `resolve_kernel` and
  `openai.APIConnectionError` into `SystemExit` messages (e.g.
  `no server at http://127.0.0.1:1/v1; start one with make serve`); a test for each.

## Low priority

Real but minor. They can be batched into one or two cleanup PRs.

- **`args=` goes through the shell unquoted** (`Makefile:23`). `make ask args="is 3 > 2"`
  writes the answer to a file named `2`, an apostrophe is a shell syntax error, and make expands
  `$`. Fix: say so in `make help`'s epilog and next to the README's `make ask` examples (calling
  the launcher directly avoids it).
- **`make bench` can't pass engine args** (launcher:340, `cmd_bench` at 191). `bench.py` takes
  `--engine-args`, but the launcher's `bench` neither declares nor forwards it, so
  `make bench args="--define MV_NR=2"` is rejected. Fix: a repeatable `--define` on `bench`
  (as on `bench-ours`), forwarded as `--engine-args`.
- **`make test` may silently skip the MLX tests** (`cmd_test`, launcher:181). It runs pytest in
  the default environment, so on a venv without `mlx-lm` every `surfaces/tests/test_mlx_e2e.py`
  test is skipped, including the MLX threading regression tests. Fix: run pytest with
  `uv run --extra mlx`, or document it.
- **Examples that go stale or break rules.** `--kernel 20261024` (README:68, `Makefile:6`,
  launcher:374) matches several folders as soon as a second kernel is created that day.
  `metal_llama_q8` (README:72, launcher:355) contains the model name, against CLAUDE.md's naming
  rule. Fix: `--kernel 20261024-1620` and `metal_q8`.
- **`serve-llamacpp` without a llama.cpp build** (launcher:138) prints cmake's raw
  `.../deps/llama.cpp/build is not a directory`. Fix: check for the build dir and say
  "run `make setup` first", like the GGUF check above it.
- **A checkout path with a space breaks every target** (`Makefile:10-11`, unquoted `$(HERE)`).
  Fix: `uv run --project . ios_kernels_launcher.py` (recipes already run in the Makefile's dir).
- **Relative paths under `make -C ios_kernels`** (`--data-dir`, `--kernel`, `new-kernel --from`)
  resolve against `ios_kernels/`, not the caller's directory (launcher:406). Fix: document it.
- **The first `make` looks hung** (`Makefile:11`): `uv run --quiet` hides the initial venv
  creation and download. Fix: drop `--quiet`, or run `help` without syncing the project.
- **`$DATA_DIR` is no longer documented** (README:22, 65 mention only `--data-dir`), but the
  direct entry points (`uv run openai-server`, `benchmarking/*.py`) only understand the env var,
  and `--data-dir`'s help shows `<repo>/.context` even when `$DATA_DIR` is set.
- **Stale numbers in working notes:** the test count at PLAN.md:253 and the timings at
  `surfaces/DESIGN_AND_LIMITATIONS.md:130`.
- **`MLXRunner.close()` doesn't join its thread** (`surfaces/runners/mlx_runner.py:76`);
  calling `stream()` after `close()` blocks forever, since no worker is left to run the job.
- **Two `new-kernel`s in the same minute** tie on `list_kernels`' sort key
  (`surfaces/paths.py:20`), so the default is arbitrary and `new-kernel`'s
  "it is now the default kernel" (launcher:274) can be wrong.
- **Signal forwarding reaches only the launcher's direct child.** A child that doesn't pass the
  signal on leaves its own children running: `swift-driver` under `bench-ours`' build step, and the
  processes under `cmake --build`. They finish on their own (they aren't servers).
- **`clean` walks all of `.venv`** before filtering out its `__pycache__` dirs (launcher:283).
- **`cli-chat` shows `0.0 tok/s` instead of `- tok/s`** when the rate is unknown but more
  than one token was reported (`surfaces/cli_client/chat.py:41-42`).

## Not yet checked

- Whether `test_mlx_from_other_threads` and `test_mlx_through_server`
  (`surfaces/tests/test_mlx_e2e.py`) actually fail on the pre-fix, single-threaded `MLXRunner`,
  i.e. whether they guard the fix they were written for.
