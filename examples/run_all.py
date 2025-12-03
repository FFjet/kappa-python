"""Run all Python example scripts."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

EXAMPLE_ROOT = Path(__file__).resolve().parent


def main() -> None:
    scripts = [
        EXAMPLE_ROOT / "basic" / "basicTest.py",
        EXAMPLE_ROOT / "particles" / "particleTest.py",
        EXAMPLE_ROOT / "particles" / "dumpspectrumTest.py",
        EXAMPLE_ROOT / "interaction" / "vssTest.py",
        EXAMPLE_ROOT / "mixtures" / "mixture-sts-shear-bulk-thermal.py",
        EXAMPLE_ROOT / "mixtures" / "mixture-sts-shear.py",
        EXAMPLE_ROOT / "mixtures" / "transport_coefficients_air5.py",
        EXAMPLE_ROOT / "approximations" / "cvibr_sts.py",
        EXAMPLE_ROOT / "approximations" / "cvrot_sts.py",
        EXAMPLE_ROOT / "approximations" / "cvtr_sts.py",
        EXAMPLE_ROOT / "approximations" / "cvibr_multit.py",
    ]
    for script in scripts:
        print(f"Running {script} ...")
        result = subprocess.run([sys.executable, str(script)], check=False)
        if result.returncode != 0:
            raise SystemExit(f"{script} failed with exit code {result.returncode}")


if __name__ == "__main__":
    main()
