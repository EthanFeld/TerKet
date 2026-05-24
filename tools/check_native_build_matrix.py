"""Run native build checks with native enabled and TERKET_DISABLE_NATIVE=1."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_build(label: str, *, disable_native: bool, root: Path) -> Path:
    build_temp = root / label / "temp"
    build_lib = root / label / "lib"
    env = os.environ.copy()
    if disable_native:
        env["TERKET_DISABLE_NATIVE"] = "1"
    else:
        env.pop("TERKET_DISABLE_NATIVE", None)
    subprocess.run(
        [
            sys.executable,
            "setup.py",
            "build_ext",
            "--build-temp",
            str(build_temp),
            "--build-lib",
            str(build_lib),
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
    )
    return build_lib


def _native_outputs(build_lib: Path) -> list[Path]:
    return sorted(build_lib.glob("terket/_schur_native*"))


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="terket-native-build-") as tmp:
        root = Path(tmp)
        disabled_lib = _run_build("disabled", disable_native=True, root=root)
        disabled_outputs = _native_outputs(disabled_lib)
        if disabled_outputs:
            raise SystemExit(f"TERKET_DISABLE_NATIVE=1 produced native outputs: {disabled_outputs}")

        enabled_lib = _run_build("enabled", disable_native=False, root=root)
        enabled_outputs = _native_outputs(enabled_lib)
        if not enabled_outputs:
            raise SystemExit("Native-enabled build produced no _schur_native extension.")

    build_dir = REPO_ROOT / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    print("native build matrix: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
