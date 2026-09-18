"""Create publication-ready plots from results/raw_results.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results(path: Path) -> np.ndarray:
    """Load and validate the experiment schema."""

    required = {
        "distance",
        "physical_error_rate",
        "shots",
        "logical_errors",
        "logical_error_rate",
        "logical_error_lower",
        "logical_error_upper",
        "physical_qubits",
        "noise_model",
        "qec",
    }
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if data.dtype.names is None or not required.issubset(data.dtype.names):
        raise ValueError(f"{path} must contain columns: {sorted(required)}")
    return data


def qec_mask(data: np.ndarray) -> np.ndarray:
    """Handle the boolean values written by csv.DictWriter."""

    return data["qec"] if data["qec"].dtype.kind == "b" else data["qec"] == "True"


def style_axes(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    ax.figure.tight_layout()


def _error_bars(subset: np.ndarray) -> np.ndarray:
    return np.vstack(
        (
            subset["logical_error_rate"] - subset["logical_error_lower"],
            subset["logical_error_upper"] - subset["logical_error_rate"],
        )
    )


def plot_physical_vs_logical(data: np.ndarray, output: Path, noise_model: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    encoded = data[qec_mask(data) & (data["noise_model"] == noise_model)]
    for distance in np.unique(encoded["distance"]):
        subset = encoded[encoded["distance"] == distance]
        order = np.argsort(subset["physical_error_rate"])
        ax.loglog(
            subset["physical_error_rate"][order],
            np.maximum(subset["logical_error_rate"][order], 0.5 / subset["shots"][order]),
            "o-",
            label=f"d = {int(distance)}",
        )
        ax.errorbar(
            subset["physical_error_rate"][order],
            np.maximum(subset["logical_error_rate"][order], 0.5 / subset["shots"][order]),
            yerr=_error_bars(subset[order]),
            fmt="none",
            capsize=3,
        )
    style_axes(ax, f"Logical error rate ({noise_model} noise)", "Physical error rate p", "Logical error rate $P_L$")
    fig.savefig(output, dpi=300)
    plt.close(fig)


def plot_resource_overhead(data: np.ndarray, output: Path, noise_model: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    encoded = data[qec_mask(data) & (data["noise_model"] == noise_model)]
    distances = np.unique(encoded["distance"])
    physical_qubits = [
        encoded[encoded["distance"] == distance]["physical_qubits"][0]
        for distance in distances
    ]
    ax.plot(distances, physical_qubits, "o-", label=noise_model)
    style_axes(ax, "Physical-qubit overhead", "Code distance d", "Physical qubits")
    fig.savefig(output, dpi=300)
    plt.close(fig)


def plot_qec_comparison(data: np.ndarray, output: Path, noise_model: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    selected = data[data["noise_model"] == noise_model]
    for distance in np.unique(selected["distance"]):
        subset = selected[selected["distance"] == distance]
        order = np.argsort(subset["physical_error_rate"])
        encoded_point = (
            bool(subset["qec"][0])
            if subset["qec"].dtype.kind == "b"
            else subset["qec"][0] == "True"
        )
        label = f"d = {int(distance)}" if encoded_point else "No QEC"
        ax.errorbar(
            subset["physical_error_rate"][order],
            np.maximum(subset["logical_error_rate"][order], 0.5 / subset["shots"][order]),
            yerr=_error_bars(subset[order]),
            fmt="o-",
            label=label,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    style_axes(ax, f"QEC versus no-QEC ({noise_model} noise)", "Physical error rate p", "Logical error rate $P_L$")
    fig.savefig(output, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/raw_results.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures"))
    parser.add_argument("--noise-model", default="depolarizing")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_results(args.input)
    plot_physical_vs_logical(data, args.output_dir / "physical_vs_logical.png", args.noise_model)
    plot_resource_overhead(data, args.output_dir / "distance_vs_physical_qubits.png", args.noise_model)
    plot_qec_comparison(data, args.output_dir / "qec_vs_no_qec.png", args.noise_model)
    print(f"Saved figures to {args.output_dir}")


if __name__ == "__main__":
    main()
