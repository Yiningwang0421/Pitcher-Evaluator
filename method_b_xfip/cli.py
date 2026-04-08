from __future__ import annotations

import argparse
import subprocess
import sys
from typing import List


def _run_module(module_name: str, forwarded_args: List[str]) -> None:
    cmd = [sys.executable, "-m", module_name, *forwarded_args]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _clean_forwarded(args: List[str]) -> List[str]:
    # Allow optional "--" separator for passthrough style invocations.
    return [arg for arg in args if arg != "--"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified Method B command entrypoint."
    )
    parser.add_argument(
        "command",
        choices=["build", "train", "full"],
        help="build: data pipeline, train: Method B training, full: build + train flag",
    )
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the underlying script.",
    )

    parsed = parser.parse_args()
    forwarded = _clean_forwarded(parsed.args)

    if parsed.command == "build":
        _run_module("method_b_xfip.scripts.pipeline", forwarded)
        return

    if parsed.command == "train":
        _run_module("method_b_xfip.scripts.train_method_b_xfip", forwarded)
        return

    # full: run pipeline and force Method B training path.
    _run_module("method_b_xfip.scripts.pipeline", ["--train-method-b", *forwarded])


if __name__ == "__main__":
    main()
