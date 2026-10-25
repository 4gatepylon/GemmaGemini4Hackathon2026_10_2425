"""Build the profiling tools (every tools/<name>/main.swift) into profiling/.build/.

    uv run profiling/build.py        # from ios_kernels/

Each tool is compiled together with lib/*.swift. Kernels are compiled at runtime by the
tools, so no Metal toolchain is needed.
"""

import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    out = HERE / ".build"
    out.mkdir(exist_ok=True)
    lib = sorted(str(p) for p in (HERE / "lib").glob("*.swift"))
    for main_swift in sorted((HERE / "tools").glob("*/main.swift")):
        tool = main_swift.parent.name
        # absolute paths, so #filePath (used to find the repo's .context/) is absolute
        subprocess.run(["swiftc", "-O", "-o", str(out / tool), *lib, str(main_swift)], check=True)
        print(f"built {out / tool}")


if __name__ == "__main__":
    main()
