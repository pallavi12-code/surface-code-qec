"""Plot the results of the rotated surface-code experiment matrix."""

from __future__ import annotations

import csv
from pathlib import Path
import matplotlib.pyplot as plt


def load_results(path: Path) -> list[dict]:
    results = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                "distance": int(row["distance"]),
                "physical_error_rate": float(row["physical_error_rate"]),
                "shots": int(row["shots"]),
                "logical_errors": int(row["logical_errors"]),
                "logical_error_rate": float(row["logical_error_rate"]),
            })
    return results


def main() -> None:
    csv_path = Path("results/raw_results.csv")
    if not csv_path.exists():
        print(f"Error: {csv_path} not found. Run run_experiments.py first.")
        return

    results = load_results(csv_path)

    # Group data by distance
    data_by_distance = {}
    for r in results:
        d = r["distance"]
        if d not in data_by_distance:
            data_by_distance[d] = {"p": [], "pl": []}
        data_by_distance[d]["p"].append(r["physical_error_rate"])
        data_by_distance[d]["pl"].append(r["logical_error_rate"])

    # Create output directory for figures
    figures_dir = Path("results/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Graph 1: Physical Error Rate vs Logical Error Rate (Log-Log Scale)
    plt.figure(figsize=(8, 6))
    for d, data in sorted(data_by_distance.items()):
        plt.plot(data["p"], data["pl"], marker="o", label=f"Distance d={d}")

    plt.xlabel("Physical Error Rate (p)")
    plt.ylabel("Logical Error Rate (PL)")
    plt.title("Surface Code QEC: Physical vs. Logical Error Rate")
    plt.yscale("log")
    plt.xscale("log")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    
    output_file = figures_dir / "physical_vs_logical_error_rate.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Successfully generated graph and saved to {output_file}")


if __name__ == "__main__":
    main()