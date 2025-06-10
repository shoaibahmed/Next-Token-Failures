"""
Plot rank-histogram statistics produced by `train.py` (saved as JSON files
in `rank_eval_outputs/`).  Each JSON looks like::

    {
        "rank_histogram": {
            "2": {"1": 123, "2": 456, ...},
            "4": {...},
        },
        "total_counts": {"2": 10000, "4": 10000},
        "run_name": "graph_multihead_gpt_...",
        "timestamp": "..."
    }

The script creates, for each head size:
1.  A bar plot of the (normalised) frequency of every rank.
2.  A CDF showing the cumulative fraction of target tokens contained in
    the top-k logits.

All plots are stored in `rank_eval_plots/<run_name>/`.
"""

import json
import os
from glob import glob
from typing import Dict

import matplotlib.pyplot as plt


def load_histograms(json_file: str):
    with open(json_file, "r") as f:
        data = json.load(f)

    rank_hist: Dict[str, Dict[str, int]] = data["rank_histogram"]
    total_counts: Dict[str, int] = data["total_counts"]
    coverage_lists: Dict[str, list] = data.get("coverage_lists", {})
    run_name: str = data.get("run_name", os.path.basename(json_file))

    # Convert keys back to int for convenience
    rank_hist = {int(hs): {int(r): int(c) for r, c in hist.items()} for hs, hist in rank_hist.items()}
    total_counts = {int(hs): int(cnt) for hs, cnt in total_counts.items()}
    # Convert coverage list keys to int
    coverage_lists = {int(hs): lst for hs, lst in coverage_lists.items()}

    return run_name, rank_hist, total_counts, coverage_lists


def plot_histograms(run_name: str,
                    rank_hist: Dict[int, Dict[int, int]],
                    total_counts: Dict[int, int],
                    coverage_lists: Dict[int, list]):
    output_dir = os.path.join("rank_eval_plots", run_name)
    os.makedirs(output_dir, exist_ok=True)

    for head_size, hist in rank_hist.items():
        counts = [hist.get(r, 0) for r in range(1, max(hist.keys()) + 1)]
        probs = [c / total_counts[head_size] for c in counts]

        # Bar plot of probability mass per rank
        plt.figure(figsize=(10, 4))
        plt.bar(range(1, len(probs) + 1), probs, color="steelblue")
        plt.xlabel("Rank")
        plt.ylabel("Probability of target token at rank")
        plt.title(f"Run: {run_name} – Head size {head_size}\nProbability mass vs. rank")
        plt.yscale("log")
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"rank_prob_head_{head_size}.pdf")
        plt.savefig(out_path)
        plt.close()

        # Cumulative distribution (top-k coverage)
        cdf = [sum(probs[:k]) for k in range(1, len(probs) + 1)]
        plt.figure(figsize=(10, 4))
        plt.plot(range(1, len(cdf) + 1), cdf, marker="o")
        plt.xlabel("k (top-k)")
        plt.ylabel("Cumulative fraction of target tokens")
        plt.title(f"Run: {run_name} – Head size {head_size}\nCumulative coverage versus k")
        plt.grid(True, which="both", linestyle="--", alpha=0.4)
        plt.ylim(0, 1.05)
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"rank_cdf_head_{head_size}.pdf")
        plt.savefig(out_path)
        plt.close()

        # Histogram of per-sequence acceptance percentages
        if head_size in coverage_lists and coverage_lists[head_size]:
            plt.figure(figsize=(6, 4))
            vals = coverage_lists[head_size]
            plt.hist([v * 100 for v in vals], bins=20, color="darkorange", edgecolor="k", alpha=0.8)
            plt.xlabel("% of target tokens accepted (per sequence)")
            plt.ylabel("Count")
            plt.title(f"Run: {run_name} – Head {head_size}\nDistribution of acceptance %")
            plt.tight_layout()
            out_path = os.path.join(output_dir, f"accept_hist_head_{head_size}.pdf")
            plt.savefig(out_path)
            plt.close()


def main():
    json_files = glob(os.path.join("rank_eval_outputs", "*_rank_hist.json"))
    if not json_files:
        print("No JSON files found in 'rank_eval_outputs/'.  Run evaluation first.")
        return

    for jf in json_files:
        print("Processing", jf)
        run_name, rank_hist, total_counts, coverage_lists = load_histograms(jf)
        plot_histograms(run_name, rank_hist, total_counts, coverage_lists)


if __name__ == "__main__":
    main()
