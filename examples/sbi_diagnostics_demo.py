"""Tiny SBI diagnostics demo with precomputed posterior samples.

This does not train a neural network.  It shows the diagnostic API on a toy
posterior sample cube with the same shape produced by CompoSED SBI estimators:

    (n_objects, n_posterior_samples, n_parameters)
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/composed_mplconfig")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inftools.diagnostics import run_sbi_diagnostics


def main() -> None:
    rng = np.random.default_rng(12)
    n_objects = 64
    n_samples = 512

    posterior_center = np.column_stack(
        [
            rng.uniform(0.0, 3.0, size=n_objects),
            rng.normal(10.0, 0.4, size=n_objects),
        ]
    )

    posterior_width = np.array([0.12, 0.18])
    theta_true = posterior_center + rng.normal(scale=posterior_width, size=(n_objects, 2))
    samples = posterior_center[:, None, :] + rng.normal(scale=posterior_width, size=(n_objects, n_samples, 2))

    output_dir = Path("outputs/sbi_diagnostics_demo")
    result = run_sbi_diagnostics(
        posterior_samples=samples,
        theta_true=theta_true,
        theta_names=["z", "log10_mass"],
        output_dir=output_dir,
        make_plots=True,
        seed=12,
    )

    print(f"Wrote diagnostics to {output_dir}")
    print("Mean empirical coverage:")
    for level, coverage in zip(result["coverage"]["levels"], result["coverage"]["mean_coverage"]):
        print(f"  {level:5.2f}: {coverage:5.2f}")


if __name__ == "__main__":
    main()
