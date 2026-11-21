#!/usr/bin/env python3
"""
Standalone script: 1x2 figure of "distribution of local dimension" histograms.

  left  : two datasets (n1=10 in blue, n1=140 in orange), N=600 runs each
  right : same kind of histogram, same bins, but fewer runs -> lower bars,
          with the two datasets still clearly separated

Synthetic data only. The mean/min/max shown in the boxes are computed from the
generated samples, so they always match what is drawn.

Usage:  python plot_histograms_local_dim.py
Tweak:  everything you probably want to change is in the CONFIG block.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# =============================================================================
# CONFIG  (edit me)
# =============================================================================
SEED = 1
FIGSIZE = (14, 5.8)
DPI = 150
OUT_FILE = "histograms_local_dim.png"   # set to None to only show the figure

COLOR_A = "tab:blue"
COLOR_B = "tab:orange"
COLOR_ADAM = "red"
COLOR_SGD = "green"
LABEL_A = r"$n_1=10$"
LABEL_B = r"$n_1=140$"
LABEL_ADAM = "Average for Adam_Uniform"
LABEL_SGD = "Average for SGD_Uniform"

# Total number of runs N in each panel, split equally between the two datasets
# (as in your figure: 300 + 300 = 600). Right panel: fewer runs -> lower bars.
N_LEFT = 600
N_RIGHT = 600

TITLES = [f"Distribution of local dimension across (N={N_LEFT}) runs at initialisation",
          f"Distribution of local dimension across (N={N_RIGHT}) runs at critical point"]

BIN_EDGES = np.arange(30, 166, 6.5)     # same bins in both panels
SHARE_Y = True                           # True => heights directly comparable
HEADROOM = 1.5                           # y-limit = HEADROOM * tallest bar

# Vertical "average" lines (x position) for each panel: (Adam, SGD)
VLINES = [(112, 102), (108, 98)]

# --- Legend / annotations ---------------------------------------------------
# "shared_bottom" : one legend under both panels (recommended)
# "per_axes"      : one legend inside each panel
# "shared_top"    : one legend above both panels
LEGEND_MODE = "shared_bottom"
LEGEND_FONTSIZE = 11
LEGEND_NCOL = 4
LEGEND_LOC_PER_AXES = ["center", "center"]   # only for "per_axes"
BOTTOM_MARGIN = 0.12                         # room reserved for shared legend
STATS_FONTSIZE = 10                          # mean/min/max boxes inside axes
STATS_POS_A = (0.02, 0.97)                   # axes fraction, anchored top-left
STATS_POS_B = (0.98, 0.97)                   # axes fraction, anchored top-right


# =============================================================================
# SYNTHETIC DATA
# =============================================================================
def sample_a(n, rng, mean=63.8, sd=11.5, lo=32, hi=90):
    """Roughly symmetric (bell) dataset, integer-valued, clipped to [lo, hi]."""
    return np.clip(np.rint(rng.normal(mean, sd, n)), lo, hi)


def sample_b(n, rng, top=158, shape=3.0, scale=4.0, lo=123):
    """Left-skewed dataset piling up just below `top`."""
    return np.clip(np.rint(top - rng.gamma(shape, scale, n)), lo, top)


def make_panel_data(n, rng, shift=0.0):
    a = sample_a(n // 2, rng, mean=63.8 + shift)
    b = sample_b(n // 2, rng, top=158 + shift)
    return a, b


# =============================================================================
# PLOTTING
# =============================================================================
def stats_text(x):
    return (f"mean: {x.mean():7.2f}\n"
            f"min:  {x.min():7.0f}\n"
            f"max:  {x.max():7.0f}")


def draw_panel(ax, a, b, vlines, title):
    ax.hist(a, bins=BIN_EDGES, color=COLOR_A, alpha=0.85, edgecolor="k")
    ax.hist(b, bins=BIN_EDGES, color=COLOR_B, alpha=0.85, edgecolor="k")
    ax.axvline(vlines[0], color=COLOR_ADAM, ls="--", lw=1.6)
    ax.axvline(vlines[1], color=COLOR_SGD, ls="--", lw=1.6)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("local dimension", fontsize=12)
    ax.set_xlim(BIN_EDGES[0] - 3, BIN_EDGES[-1] + 3)

    for x, color, pos, ha in ((a, COLOR_A, STATS_POS_A, "left"),
                              (b, COLOR_B, STATS_POS_B, "right")):
        ax.text(*pos, stats_text(x), transform=ax.transAxes, ha=ha, va="top",
                family="monospace", fontsize=STATS_FONTSIZE,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=color,
                          lw=1.8, alpha=0.95))


def legend_handles():
    return [
        Patch(facecolor=COLOR_A, edgecolor="k", alpha=0.85, label=LABEL_A),
        Patch(facecolor=COLOR_B, edgecolor="k", alpha=0.85, label=LABEL_B),
        Line2D([], [], color=COLOR_ADAM, ls="--", lw=1.6, label=LABEL_ADAM),
        Line2D([], [], color=COLOR_SGD, ls="--", lw=1.6, label=LABEL_SGD),
    ]


def main():
    rng = np.random.default_rng(SEED)

    # right panel: slightly different distribution (shifted a bit), fewer runs
    a_l, b_l = make_panel_data(N_LEFT, rng)
    a_r, b_r = make_panel_data(N_RIGHT, rng, shift=-3.0)
    data = [(a_l, b_l), (a_r, b_r)]

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, sharey=SHARE_Y)
    for ax, (a, b), vl, title in zip(axes, data, VLINES, TITLES):
        draw_panel(ax, a, b, vl, title)
    axes[0].set_ylabel("number of runs", fontsize=12)
    if not SHARE_Y:
        axes[1].set_ylabel("number of runs", fontsize=12)

    # y-limits with headroom for the stats boxes
    tallest = [max(np.histogram(x, BIN_EDGES)[0].max() for x in pair)
               for pair in data]
    if SHARE_Y:
        axes[0].set_ylim(0, HEADROOM * max(tallest))
    else:
        for ax, t in zip(axes, tallest):
            ax.set_ylim(0, HEADROOM * t)

    for ax in axes:
        ax.grid(alpha=0.25, axis="y")
        ax.set_axisbelow(True)

    # legend + layout
    h = legend_handles()
    if LEGEND_MODE == "shared_bottom":
        fig.tight_layout(rect=(0, BOTTOM_MARGIN, 1, 1))
        fig.legend(handles=h, ncol=LEGEND_NCOL, loc="lower center",
                   bbox_to_anchor=(0.5, 0.01), fontsize=LEGEND_FONTSIZE)
    elif LEGEND_MODE == "shared_top":
        fig.tight_layout(rect=(0, 0, 1, 1 - BOTTOM_MARGIN))
        fig.legend(handles=h, ncol=LEGEND_NCOL, loc="upper center",
                   bbox_to_anchor=(0.5, 0.99), fontsize=LEGEND_FONTSIZE)
    elif LEGEND_MODE == "per_axes":
        fig.tight_layout()
        for ax, loc in zip(axes, LEGEND_LOC_PER_AXES):
            ax.legend(handles=h, loc=loc, fontsize=LEGEND_FONTSIZE - 2,
                      framealpha=0.9)
    else:
        raise ValueError(f"Unknown LEGEND_MODE: {LEGEND_MODE}")

    if OUT_FILE:
        fig.savefig(OUT_FILE, dpi=DPI, bbox_inches="tight")
        print(f"saved -> {OUT_FILE}")
    plt.show()


if __name__ == "__main__":
    main()