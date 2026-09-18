"""Run the standard rotated surface-code experiment matrix."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from surface_code_simulator import NOISE_MODELS, run_matrix


DISTANCES = (3, 5, 7, 9)
PHYSICAL_ERROR_RATES = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05)
DEFAULT_SHOTS = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS)
    parser.add_argument("--low-error-shots", type=int, default=100_000)
    parser.add_argument("--low-error-cutoff", type=float, default=0.002)
    parser.add_argument("--noise-model", choices=NOISE_MODELS, default="depolarizing")
    parser.add_argument("--all-noise-models", action="store_true")
    parser.add_argument("--without-distance-9", action="store_true")
    parser.add_argument("--without-no-qec", action="store_true")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output", type=Path, default=Path("results/raw_results.csv"))
    return parser.parse_args()


def write_results(path: Path, results: list) -> None:
    """Write stable, analysis-friendly CSV output."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "distance",
                "physical_error_rate",
                "shots",
                "logical_errors",
                "logical_error_rate",
                "logical_error_lower",
                "logical_error_upper",
                "physical_qubits",
                "detectors",
                "observables",
                "noise_model",
                "qec",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "distance": result.distance,
                    "physical_error_rate": result.physical_error_rate,
                    "shots": result.shots,
                    "logical_errors": result.logical_errors,
                    "logical_error_rate": result.logical_error_rate,
                    "logical_error_lower": result.confidence_interval[0],
                    "logical_error_upper": result.confidence_interval[1],
                    "physical_qubits": result.physical_qubits,
                    "detectors": result.detectors,
                    "observables": result.observables,
                    "noise_model": result.noise_model,
                    "qec": result.qec,
                }
            )


def main() -> None:
    args = parse_args()
    if args.shots < 10_000 or args.low_error_shots < 10_000:
        raise ValueError("at least 10,000 shots per point are required")

    distances = DISTANCES[:-1] if args.without_distance_9 else DISTANCES
    rates_and_shots = [
        (rate, args.low_error_shots if rate <= args.low_error_cutoff else args.shots)
        for rate in PHYSICAL_ERROR_RATES
    ]
    models = NOISE_MODELS if args.all_noise_models else (args.noise_model,)
    results = []
    for model_index, model in enumerate(models):
        for rate_index, (rate, shots) in enumerate(rates_and_shots):
            results.extend(
                run_matrix(
                    distances,
                    (rate,),
                    shots,
                    None
                    if args.seed is None
                    else args.seed + model_index * 10_000 + rate_index * 100,
                    model,
                    not args.without_no_qec,
                )
            )
    print("distance, physical_error_rate, shots, logical_errors, logical_error_rate")
    for result in results:
        print(
            f"{result.distance}, {result.physical_error_rate:.3g}, "
            f"{result.shots}, {result.logical_errors}, "
            f"{result.logical_error_rate:.6g} "
            f"[{result.confidence_interval[0]:.6g}, {result.confidence_interval[1]:.6g}]"
        )
    write_results(args.output, results)
    print(f"\nSaved raw results to {args.output}")


if __name__ == "__main__":
    main()
