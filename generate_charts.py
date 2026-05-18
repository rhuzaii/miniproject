"""
Generate all paper charts from collected result CSVs.
Run: python generate_charts.py
Saves all figures to tests/results/
"""

import csv, os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS = "tests/results"


def load_csv(name):
    with open(os.path.join(RESULTS, name), newline="") as f:
        return list(csv.DictReader(f))


# ── Fig 4 — End-to-End Latency Bar Chart ──────────────────────────────────────

def fig4_latency():
    rows    = load_csv("table7_latency.csv")
    stages  = [r["Stage"] for r in rows]
    medians = [float(r["Median (ms)"]) for r in rows]
    p95s    = [float(r["P95 (ms)"]) for r in rows]

    x, w = np.arange(len(stages)), 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    b1 = ax.bar(x - w/2, medians, w, label="Median",          color="#4C9BE8", zorder=3)
    b2 = ax.bar(x + w/2, p95s,    w, label="95th Percentile", color="#E87B4C", zorder=3)

    ax.set_xlabel("Pipeline Stage", fontsize=12)
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title("Fig 4 — End-to-End Latency Breakdown\n(Gesture-Based Alexa Control System)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace(" → ", "\n→ ") for s in stages], fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.set_ylim(0, max(p95s) * 1.25)

    for bar in list(b1) + list(b2):
        ax.annotate(f"{bar.get_height():.0f}",
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out = os.path.join(RESULTS, "fig4_latency_chart.png")
    plt.savefig(out, dpi=180)
    print(f"Saved → {out}")
    plt.close()


# ── Fig 5 — Gesture Accuracy Heatmap ──────────────────────────────────────────

def fig5_gesture_accuracy():
    rows = load_csv("table4_gesture_accuracy.csv")

    gestures  = ["Thumbs Up", "Thumbs Down", "Open Palm", "Closed Fist", "Peace / V Sign", "Three Fingers"]
    lightings = ["Bright", "Normal", "Low-Light"]

    data = np.zeros((len(gestures), len(lightings)))
    for r in rows:
        gi = gestures.index(r["Gesture"])
        li = lightings.index(r["Lighting"])
        data[gi][li] = float(r["Accuracy (%)"])

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(data, cmap="RdYlGn", vmin=50, vmax=100, aspect="auto")

    ax.set_xticks(range(len(lightings)));  ax.set_xticklabels(lightings, fontsize=11)
    ax.set_yticks(range(len(gestures)));   ax.set_yticklabels(gestures, fontsize=11)
    ax.set_title("Fig 5 — Gesture Recognition Accuracy (%) by Lighting Condition", fontsize=12)

    for i in range(len(gestures)):
        for j in range(len(lightings)):
            val = data[i][j]
            color = "black" if val > 70 else "white"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                    fontsize=12, fontweight="bold", color=color)

    plt.colorbar(im, ax=ax, label="Accuracy (%)")
    plt.tight_layout()
    out = os.path.join(RESULTS, "fig5_gesture_accuracy_heatmap.png")
    plt.savefig(out, dpi=180)
    print(f"Saved → {out}")
    plt.close()


# ── Fig 6 — Stability Buffer Trade-off ────────────────────────────────────────

def fig6_stability_buffer():
    rows     = load_csv("table5_stability_buffer.csv")
    ns       = [int(r["Buffer Size (N)"]) for r in rows]
    ftr      = [float(r["False Trigger Rate (%)"]) for r in rows]
    latency  = [float(r["Avg Confirmation Latency (ms)"]) for r in rows]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    color1 = "#E74C3C"
    ax1.set_xlabel("Buffer Size (N frames)", fontsize=12)
    ax1.set_ylabel("False Trigger Rate (%)", color=color1, fontsize=12)
    ax1.plot(ns, ftr, color=color1, marker="o", linewidth=2, markersize=8, label="False Trigger Rate")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(0, max(ftr) * 1.3)

    ax2 = ax1.twinx()
    color2 = "#2980B9"
    ax2.set_ylabel("Avg Confirmation Latency (ms)", color=color2, fontsize=12)
    ax2.plot(ns, latency, color=color2, marker="s", linewidth=2, markersize=8,
             linestyle="--", label="Latency")
    ax2.tick_params(axis="y", labelcolor=color2)

    # Mark the chosen buffer size N=6
    ax1.axvline(x=6, color="green", linestyle=":", linewidth=2, alpha=0.7)
    ax1.text(6.2, max(ftr)*0.8, "N=6\n(chosen)", color="green", fontsize=10)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=10)

    ax1.set_title("Fig 6 — Stability Buffer: False Trigger Rate vs Confirmation Latency", fontsize=12)
    ax1.set_xticks(ns)
    ax1.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    out = os.path.join(RESULTS, "fig6_stability_buffer.png")
    plt.savefig(out, dpi=180)
    print(f"Saved → {out}")
    plt.close()


# ── Run all ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(RESULTS, exist_ok=True)
    fig4_latency()
    fig5_gesture_accuracy()
    fig6_stability_buffer()
    print("\nAll charts saved to tests/results/")
