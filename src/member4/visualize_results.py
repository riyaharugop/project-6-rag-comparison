
"""Person 4 - Generate retrieval comparison charts."""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_FILE = PROJECT_ROOT / "results" / "retrieval_metrics.csv"
CHARTS_DIR = PROJECT_ROOT / "results" / "charts"


def main():
    data = pd.read_csv(RESULTS_FILE)

    summary = (
        data.groupby(["chunking_strategy", "embedding_model"])
        [
            [
                "context_recall",
                "context_precision",
                "retrieval_latency_seconds",
            ]
        ]
        .mean()
        .reset_index()
    )

    summary["configuration"] = (
        summary["chunking_strategy"]
        .str.replace("_", " ")
        + " + "
        + summary["embedding_model"].str.split("/").str[-1]
    )

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    charts = [
        ("context_recall", "Average Context Recall", "Recall"),
        ("context_precision", "Average Context Precision", "Precision"),
        (
            "retrieval_latency_seconds",
            "Average Retrieval Latency",
            "Latency (seconds)",
        ),
    ]

    for metric, title, ylabel in charts:
        plt.figure(figsize=(11, 6))
        plt.bar(summary["configuration"], summary[metric])
        plt.title(title)
        plt.xlabel("Configuration")
        plt.ylabel(ylabel)
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()

        filename = f"{metric}.png"
        plt.savefig(CHARTS_DIR / filename, dpi=300)
        plt.close()

        print(f"Saved: {CHARTS_DIR / filename}")

    summary.to_csv(
        PROJECT_ROOT / "results" / "comparison_summary.csv",
        index=False,
    )

    print("\nComparison summary saved.")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
