from __future__ import annotations

import csv
import importlib.util
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
BENCHMARKS_ROOT = REPO_ROOT / "benchmarks"
PYTHON = sys.executable

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from terket import compute_circuit_amplitude, make_circuit
from terket.head_to_head_cases import build_approximate_qft, build_approximate_qft_logical
from terket.circuit_spec import from_qiskit


def _dependencies_available(*modules: str) -> bool:
    return all(importlib.util.find_spec(module) is not None for module in modules)


def _run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(SRC_ROOT)
        if not pythonpath
        else os.pathsep.join([str(SRC_ROOT), pythonpath])
    )
    return subprocess.run(
        args,
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )


def _load_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class ApiSmokeTests(unittest.TestCase):
    def test_public_api_smoke(self):
        circuit = make_circuit(1, [("h", 0)])
        amplitude, _ = compute_circuit_amplitude(circuit, [0], [0], as_complex=True)
        self.assertAlmostEqual(abs(complex(amplitude)), math.sqrt(0.5), places=12)

    def test_approximate_qft_direct_builder_matches_qiskit_import(self):
        direct = build_approximate_qft(8)
        imported = from_qiskit(build_approximate_qft_logical(8), rz_compile_mode="dyadic")

        zero = (0,) * 8
        direct_amp, _ = compute_circuit_amplitude(direct, zero, zero, as_complex=True)
        imported_amp, _ = compute_circuit_amplitude(imported, zero, zero, as_complex=True)

        self.assertAlmostEqual(complex(direct_amp).real, complex(imported_amp).real, places=12)
        self.assertAlmostEqual(complex(direct_amp).imag, complex(imported_amp).imag, places=12)


@unittest.skipUnless(
    _dependencies_available("numpy", "psutil", "qiskit", "quimb", "cotengra"),
    "benchmark dependencies are not installed",
)
class BenchmarkCliSmokeTests(unittest.TestCase):
    def test_head_to_head_smoke_suite(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "head_to_head_smoke.csv"
            _run_command(
                [
                    PYTHON,
                    str(BENCHMARKS_ROOT / "run_benchmarks.py"),
                    "head-to-head",
                    "--suite",
                    "smoke",
                    "--repeats",
                    "1",
                    "--csv",
                    str(csv_path),
                ]
            )
            rows = _load_csv_rows(csv_path)
            self.assertEqual(
                {row["case"] for row in rows},
                {
                    "grover16",
                    "qaoa16",
                    "approximate_qft32",
                    "toffoli_ladder16",
                    "draper8",
                    "qec_repetition_magic_round8",
                },
            )

    def test_structured_showcase_smoke_suite(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "structured_smoke.csv"
            _run_command(
                [
                    PYTHON,
                    str(BENCHMARKS_ROOT / "run_benchmarks.py"),
                    "structured-showcase",
                    "--suite",
                    "smoke",
                    "--csv",
                    str(csv_path),
                ]
            )
            rows = _load_csv_rows(csv_path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["case"], "mm_hidden_shift24")


if __name__ == "__main__":
    unittest.main()
