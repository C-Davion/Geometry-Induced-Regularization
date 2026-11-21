#!/usr/bin/env python3
"""
Standalone script: 3 subplots (rank Df, rank(D), |A|) vs n1, each showing
  - the "current" curves (blue)      : median, 25-75% band, min, max
  - a "new" curve set (orange)       : same structure, slightly lower, same trends
  - the dashed line "# training samples"

All data are synthetic (digitised roughly from your figure), so you can play
with the legend layout and the styling without re-running your experiments.

Usage:  python plot_ranks_comparison.py
Tweak:  everything you probably want to change is in the CONFIG block below.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# =============================================================================
# CONFIG  (edit me)
# =============================================================================
SEED = 0
N_TRAIN = 160                      # "# training samples" horizontal line
FIGSIZE = (17, 5.2)
DPI = 150
OUT_FILE =None  # set to None to only show the figure

COLOR_OLD = "tab:blue"
COLOR_NEW = "tab:orange"
COLOR_TRAIN = "k"
LABEL_OLD = "critical point"
LABEL_NEW = "initialisation"

# How much lower the new curves are (relative gap, per subplot)
GAP = [0.03, 0.08, 0.08]
NOISE = 0.01                       # relative jitter on the new curves

# --- Legend -----------------------------------------------------------------
# "shared_bottom" : 2 compact legends under all subplots (colour = series,
#                   line style = statistic)  -> recommended for slides
# "shared_top"    : same, above the subplots
# "per_axes"      : one full legend inside each subplot (like your original)
# "outside_right" : one legend column on the right of the figure
LEGEND_MODE = "shared_bottom"
LEGEND_FONTSIZE = 11
LEGEND_NCOL_SERIES = 2             # columns of the "series" legend
LEGEND_NCOL_STATS = 5              # columns of the "statistics" legend
LEGEND_GAP = 0.02                  # horizontal gap between the 2 legends (fig. fraction)
PER_AXES_LOC = ["lower right", "lower right", "upper left"]   # used in "per_axes"
BOTTOM_MARGIN = 0.14               # room reserved for the shared legend

# =============================================================================
# SYNTHETIC "CURRENT" DATA (roughly digitised from the figure)
# =============================================================================
n1 = np.array([10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240,
               300, 360, 420, 480])

CURRENT = {
    "rank Df": dict(
        med=[113.5, 120, 129.5, 130, 144, 143.5, 147.5, 147, 150, 153.5, 153,
             155, 154, 158, 157.5, 157, 157.5],
        mn=[83, 108, 117, 118, 139, 141, 141, 138, 142, 147, 149, 151, 150,
            152, 150, 155, 153],
        mx=[120, 129, 136, 145, 150, 152, 151, 157, 156, 159, 154, 157, 158,
            159, 159, 158, 158.5],
        iqr=[6, 4, 5, 5, 3, 2.5, 3, 3, 3, 2, 2, 1.5, 1.5, 1.5, 2, 1.5, 1.5],
        title=r"$\mathrm{rank}\,\mathbf{D}f(\theta,X)$",
        ylim=(78, 163),
        cap=N_TRAIN,
    ),
    "|A|": dict(
        med=[11, 20.5, 24.5, 34.5, 42, 48.5, 49.5, 52, 57.5, 64.5, 63.5, 68,
             70, 83.5, 91, 90.5, 94],
        mn=[5, 13, 18, 24, 37, 41, 42, 45, 45, 54, 47, 58, 63, 72, 69, 72, 74],
        mx=[15, 30, 38, 45, 48, 56, 66, 67, 64, 73, 72, 78, 77, 101, 104, 96,
            104],
        iqr=[1.5, 2, 3, 3, 4, 4, 4, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7],
        title=r"$|\mathcal{A}(\theta,X)|$",
        ylim=(-3, 163),
        cap=None,
    ),
     "rank D": dict(
        med=[8.5, 16, 23.5, 29, 39.5, 44.5, 45.5, 50, 51.5, 59.5, 59.5, 65,
             66, 78, 85.5, 84.5, 90],
        mn=[5, 15, 19, 23, 36, 32, 39, 43, 40, 50, 45, 56, 57, 66, 63, 64, 71],
        mx=[9, 19, 30, 39, 42, 50, 59.5, 57, 60, 66, 67, 74, 74, 97, 97, 94,
            98],
        iqr=[1, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 7],
        title=r"$\mathrm{rank}(D)$",
        ylim=(-3, 163),
        cap=None,
    ),
}


# =============================================================================
# SYNTHETIC "NEW" DATA
# =============================================================================
def make_new(cur, gap, rng, noise=NOISE):
    """Slightly lower copy of the current stats: same trend and same bumps
    (multiplicative scaling) + a bit of jitter, with min<=q25<=med<=q75<=max
    enforced and a cap at N_TRAIN when relevant."""
    med = np.asarray(cur["med"], float)
    mn = np.asarray(cur["mn"], float)
    mx = np.asarray(cur["mx"], float)
    h = np.asarray(cur["iqr"], float)

    # gap shrinks a little when going to large n1 (curves get closer) but stays > 0
    g = gap * (1.0 - 0.3 * np.linspace(0, 1, len(med)))
    jit = lambda: 1 + noise * rng.standard_normal(len(med))

    new_med = med * (1 - g) * jit()
    new_mn = mn * (1 - 1.2 * g) * jit()
    new_mx = mx * (1 - 0.8 * g) * jit()
    new_q25 = new_med - h * (0.9 + 0.2 * rng.random(len(med)))
    new_q75 = new_med + h * (0.9 + 0.2 * rng.random(len(med)))

    # consistency: min <= q25 <= med <= q75 <= max (and cap)
    new_mn = np.minimum(new_mn, new_q25 - 0.5)
    new_mx = np.maximum(new_mx, new_q75 + 0.5)
    if cur["cap"] is not None:
        new_mx = np.minimum(new_mx, cur["cap"] - 0.5)
        new_q75 = np.minimum(new_q75, new_mx)
        new_med = np.minimum(new_med, new_q75)
    return dict(med=new_med, mn=new_mn, mx=new_mx, q25=new_q25, q75=new_q75)


def to_stats(cur):
    med = np.asarray(cur["med"], float)
    h = np.asarray(cur["iqr"], float)
    return dict(med=med, mn=np.asarray(cur["mn"], float),
                mx=np.asarray(cur["mx"], float), q25=med - h, q75=med + h)


# =============================================================================
# PLOTTING
# =============================================================================
def draw_series(ax, x, s, color):
    """Median (+markers), 25-75% band, min (dashed), max (dotted)."""
    ax.fill_between(x, s["q25"], s["q75"], color=color, alpha=0.2, lw=0)
    ax.plot(x, s["mx"], color=color, ls=":", lw=1.2)
    ax.plot(x, s["mn"], color=color, ls="--", lw=1.2)
    ax.plot(x, s["med"], color=color, ls="-", lw=1.8, marker="o", ms=5)


def stat_handles(color="0.25"):
    """Neutral handles explaining line styles (used by the shared legend)."""
    return [
        Line2D([], [], color=color, lw=1.8, marker="o", ms=5, label="median"),
        Patch(facecolor=color, alpha=0.2, label="25-75%"),
        Line2D([], [], color=color, ls="--", lw=1.2, label="min"),
        Line2D([], [], color=color, ls=":", lw=1.2, label="max"),
        Line2D([], [], color=COLOR_TRAIN, ls="--", lw=1.5, label="# training samples"),
    ]


def series_handles():
    return [
        Line2D([], [], color=COLOR_OLD, lw=3, label=LABEL_OLD),
        Line2D([], [], color=COLOR_NEW, lw=3, label=LABEL_NEW),
    ]


def add_legends(fig, axes):
    fs = LEGEND_FONTSIZE
    if LEGEND_MODE in ("shared_bottom", "shared_top"):
        bottom = LEGEND_MODE == "shared_bottom"
        y = 0.01 if bottom else 0.99
        loc = "lower " if bottom else "upper "
        # two legends side by side, centred as a block around x = 0.5
        l1 = fig.legend(handles=series_handles(), ncol=LEGEND_NCOL_SERIES,
                        loc=loc + "right", bbox_to_anchor=(0.5 - LEGEND_GAP / 2, y),
                        fontsize=fs, title="Series", title_fontsize=fs,
                        frameon=True)
        fig.legend(handles=stat_handles(), ncol=LEGEND_NCOL_STATS,
                   loc=loc + "left", bbox_to_anchor=(0.5 + LEGEND_GAP / 2, y),
                   fontsize=fs, title="Statistics", title_fontsize=fs,
                   frameon=True)
    elif LEGEND_MODE == "per_axes":
        for ax, loc in zip(axes, PER_AXES_LOC):
            ax.legend(handles=series_handles() + stat_handles(), loc=loc,
                      ncol=2, fontsize=fs - 3, framealpha=0.9, columnspacing=1.0)
    elif LEGEND_MODE == "outside_right":
        fig.legend(handles=series_handles() + stat_handles(), loc="center left",
                   bbox_to_anchor=(0.985, 0.5), fontsize=fs, frameon=True)
    else:
        raise ValueError(f"Unknown LEGEND_MODE: {LEGEND_MODE}")


def main():
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE)

    for ax, (key, cur), gap in zip(axes, CURRENT.items(), GAP):
        old = to_stats(cur)
        new = make_new(cur, gap, rng)
        draw_series(ax, n1, old, COLOR_OLD)
        draw_series(ax, n1, new, COLOR_NEW)
        ax.axhline(N_TRAIN, color=COLOR_TRAIN, ls="--", lw=1.5)
        ax.set_title(cur["title"], fontsize=14)
        ax.set_xlabel(r"$n_1$", fontsize=12)
        ax.set_ylabel("value", fontsize=12)
        ax.set_ylim(*cur["ylim"])
        ax.set_xlim(-10, 500)
        ax.grid(alpha=0.3)
    fig.suptitle("figure prototype avec données synthétiques")
    # layout: reserve space for the shared legend
    if LEGEND_MODE == "shared_bottom":
        fig.tight_layout(rect=(0, BOTTOM_MARGIN, 1, 1))
    elif LEGEND_MODE == "shared_top":
        fig.tight_layout(rect=(0, 0, 1, 1 - BOTTOM_MARGIN))
    elif LEGEND_MODE == "outside_right":
        fig.tight_layout(rect=(0, 0, 0.85, 1))
    else:
        fig.tight_layout()
    add_legends(fig, axes)
    
    
    if OUT_FILE:
        fig.savefig(OUT_FILE, dpi=DPI, bbox_inches="tight")
        print(f"saved -> {OUT_FILE}")
    plt.show()


if __name__ == "__main__":
    main()