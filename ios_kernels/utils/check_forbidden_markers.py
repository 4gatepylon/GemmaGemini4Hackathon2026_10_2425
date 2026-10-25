"""Fail if work since the base branch contains a forbidden marker (case-insensitive substring).

Checks lines added in `git diff <merge-base> HEAD`, the messages of commits in
`<merge-base>..HEAD`, and the PR title/body when $PR_TITLE / $PR_BODY are set (in CI).
Anything already on the base branch is fine.

    uv run utils/check_forbidden_markers.py [--base origin/main]
"""

import argparse
import os
import subprocess
import sys

# Built by concatenation so this file doesn't contain the markers itself.
MARKERS = ["DO NOT " + "SUBMIT", "D" + "NS"]


def find_markers(text: str) -> list[str]:
    lower = text.lower()
    return [m for m in MARKERS if m.lower() in lower]


def git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="origin/main", help="base branch ref (default: origin/main)")
    args = ap.parse_args()
    base = git("merge-base", args.base, "HEAD").strip()

    problems = []
    file = None
    for line in git("diff", "--unified=0", "--no-color", base, "HEAD").splitlines():
        if line.startswith("+++ "):
            file = line[6:] if line.startswith("+++ b/") else line[4:]
        elif line.startswith("+") and (found := find_markers(line[1:])):
            problems.append(f"{file}: added line contains {found}: {line[1:].strip()[:120]}")
    for commit in git("rev-list", f"{base}..HEAD").split():
        if found := find_markers(git("log", "-1", "--format=%B", commit)):
            problems.append(f"commit {commit[:10]} message contains {found}")
    for var in ("PR_TITLE", "PR_BODY"):
        if found := find_markers(os.environ.get(var, "")):
            problems.append(f"{var.lower().replace('_', ' ')} contains {found}")

    for p in problems:
        print(f"error: {p}", file=sys.stderr)
    if problems:
        sys.exit(1)
    print(f"no forbidden markers since {args.base}")


if __name__ == "__main__":
    main()
