#!/usr/bin/env python
"""Compare the JAX Cue port against the public Cue NumPy/PCA calculation.

This script is deliberately diagnostic rather than polished. It answers one
question: did the JAX port reproduce the public Cue emulator math for the same
input parameters?

Required local inputs
---------------------
- A clone or installation of the public Cue repository with its ``data`` folder.
- ``dill`` and ``scikit-learn`` to read Cue's pickle files.
- JAX to run the port.

The public Cue data directory is usually:

    /path/to/cue/src/cue/data

The script does not import TensorFlow. It reads the public Speculator/PCA
weights and evaluates the same calculation with NumPy and JAX.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from composed.experimental.jaxcigale.cue_port import CueJaxPort
from composed.experimental.jaxcigale.dependencies import require_jax


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cue-data-dir",
        type=Path,
        default=Path(os.environ.get("CUE_DATA_DIR", "/private/tmp/cue/src/cue/data")),
        help="Path to public Cue data directory containing speculator_*.pkl and pca_*.pkl files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experimental_cue_jax_port_comparison"),
        help="Directory for plots and numerical summary.",
    )
    return parser.parse_args()


def representative_public_cue_thetas() -> np.ndarray:
    """Three broad, in-domain Cue parameter vectors.

    Public Cue convention:

    gamma1..4, logLratio1..3, logQ, n_H, [O/H], log(N/O), log(C/O)
    """

    return np.asarray(
        [
            [2.5, 1.0, 0.2, -0.6, 2.0, 0.5, 0.4, 52.0, 100.0, -0.3, -0.134, -0.134],
            [4.0, 2.0, 1.0, 0.0, 1.0, 0.2, 0.1, 51.5, 300.0, -1.0, -0.5, -0.3],
            [8.0, 3.0, 1.5, 0.5, 4.0, 1.0, 0.8, 53.0, 30.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )


def finite_log10(values: np.ndarray) -> np.ndarray:
    return np.log10(np.maximum(np.asarray(values, dtype=float), 1e-300))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    jax, jnp = require_jax()
    print("JAX backend:", jax.default_backend())
    print("Cue data directory:", args.cue_data_dir)

    port = CueJaxPort.from_public_cue_data_dir(args.cue_data_dir)
    theta = representative_public_cue_thetas()

    continuum_wave, continuum_numpy = port.predict_continuum_native_numpy(theta)
    _, continuum_jax = port.predict_continuum_native_jax(jnp.asarray(theta))
    line_wave, line_numpy = port.predict_lines_native_numpy(theta)
    _, line_jax = port.predict_lines_native_jax(jnp.asarray(theta))

    continuum_jax = np.asarray(continuum_jax)
    line_jax = np.asarray(line_jax)

    continuum_log_diff = finite_log10(continuum_jax) - finite_log10(continuum_numpy)
    line_log_diff = finite_log10(line_jax) - finite_log10(line_numpy)

    summary = {
        "jax_backend": jax.default_backend(),
        "cue_data_dir": str(args.cue_data_dir),
        "n_theta": int(theta.shape[0]),
        "continuum_shape": list(continuum_numpy.shape),
        "line_shape": list(line_numpy.shape),
        "max_abs_log10_continuum_difference": float(np.max(np.abs(continuum_log_diff))),
        "max_abs_log10_line_difference": float(np.max(np.abs(line_log_diff))),
        "median_abs_log10_continuum_difference": float(np.median(np.abs(continuum_log_diff))),
        "median_abs_log10_line_difference": float(np.median(np.abs(line_log_diff))),
    }

    print(json.dumps(summary, indent=2))

    fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    for i in range(theta.shape[0]):
        ax = axes[i, 0]
        ax.plot(continuum_wave, continuum_numpy[i], color="black", lw=2.0, label="Cue public NumPy/PCA")
        ax.plot(continuum_wave, continuum_jax[i], color="tab:orange", lw=1.2, ls="--", label="composed JAX port")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Rest wavelength [Angstrom]")
        ax.set_ylabel("Cue native continuum output")
        ax.set_title(f"Continuum, theta #{i}")
        if i == 0:
            ax.legend(loc="best")

        ax = axes[i, 1]
        ax.scatter(line_wave, line_numpy[i], s=15, color="black", label="Cue public NumPy/PCA")
        ax.scatter(line_wave, line_jax[i], s=10, color="tab:orange", marker="x", label="composed JAX port")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Rest wavelength [Angstrom]")
        ax.set_ylabel("Cue native line output")
        ax.set_title(f"Lines, theta #{i}")
        if i == 0:
            ax.legend(loc="best")

    plot_path = args.output_dir / "cue_jax_port_spectra_comparison.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    axes[0].plot(continuum_wave, continuum_log_diff.T, lw=1.0)
    axes[0].axhline(0.0, color="black", lw=0.8)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Rest wavelength [Angstrom]")
    axes[0].set_ylabel("log10(JAX) - log10(public)")
    axes[0].set_title("Continuum residuals")

    axes[1].scatter(np.tile(line_wave, theta.shape[0]), line_log_diff.reshape(-1), s=10)
    axes[1].axhline(0.0, color="black", lw=0.8)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Rest wavelength [Angstrom]")
    axes[1].set_ylabel("log10(JAX) - log10(public)")
    axes[1].set_title("Line residuals")

    residual_path = args.output_dir / "cue_jax_port_residuals.png"
    fig.savefig(residual_path, dpi=180)
    plt.close(fig)

    np.savez(
        args.output_dir / "cue_jax_port_comparison.npz",
        theta_public=theta,
        continuum_wave_a=continuum_wave,
        continuum_public=continuum_numpy,
        continuum_jax=continuum_jax,
        line_wave_a=line_wave,
        line_public=line_numpy,
        line_jax=line_jax,
    )
    (args.output_dir / "cue_jax_port_summary.json").write_text(json.dumps(summary, indent=2))

    print("Saved:", plot_path)
    print("Saved:", residual_path)


if __name__ == "__main__":
    main()
