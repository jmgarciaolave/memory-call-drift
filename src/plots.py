"""Figures for the README and the deck. Static PNGs, one message per chart."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import OUTCOME, TARGET

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlelocation": "left",
})

SHORT = {
    "Benchmark: always the majority class": "Always predict 'lags SOX'",
    "Benchmark: follow the day-one move": "Follow the day-one move",
    "Numbers: logistic regression": "Numbers: logistic regression",
    "Numbers: gradient-boosted trees": "Numbers: boosted trees",
    "Text: neural network on call vectors": "Text: neural network",
    "Numbers + word-count tone": "Numbers + tone score",
    "Numbers + text network": "Numbers + text network",
    "Text: neural network on FinBERT embeddings": "Text: network on FinBERT",
    "Numbers + FinBERT tone": "Numbers + FinBERT tone",
}


def _colour(name):
    if name.startswith("Benchmark"):
        return GREY
    if name.startswith("Numbers +"):
        return AQUA
    return ORANGE if name.startswith("Text") else BLUE


def _save(fig, folder, name):
    fig.savefig(os.path.join(folder, name), dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_outcome(df, folder):
    """How often each company beat SOX over days 2-20."""
    labels = {"Micron": "Micron", "SK_Hynix": "SK Hynix", "Samsung": "Samsung",
              "Sandisk_Lineage": "WD / Sandisk"}
    share = df.groupby("producer")[TARGET].agg(["mean", "size"]).loc[list(labels)]
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    ypos = np.arange(len(share))[::-1]
    ax.barh(ypos, share["mean"], height=0.5, color=BLUE)
    ax.axvline(0.5, color=MUTED, lw=1, ls=(0, (3, 3)))
    ax.text(0.505, -0.47, "coin flip", color=MUTED, fontsize="small", ha="left", va="center")
    ax.set_ylim(-0.65, len(share) - 0.55)
    for yy, (v, n) in zip(ypos, share.to_numpy()):
        ax.text(v + 0.01, yy, f"{v:.0%}  ({int(n)} calls)", va="center", color=INK, fontsize="small",
                bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5})
    ax.set_yticks(ypos, [labels[p] for p in share.index])
    ax.set_xlim(0, 0.75); ax.set_xticks([0, 0.25, 0.5, 0.75], ["0%", "25%", "50%", "75%"])
    ax.tick_params(length=0); ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_title("Share of calls where the stock beat SOX over days 2 to 20")
    _save(fig, folder, "fig1_outcome_by_company.png")


def _dot_whisker(ax, names, mid, lo, hi, ref, fmt):
    ypos = np.arange(len(names))[::-1]
    for yy, n, m_, l_, h_ in zip(ypos, names, mid, lo, hi):
        c = _colour(n)
        if not (np.isnan(l_) or np.isnan(h_)):
            ax.plot([l_, h_], [yy, yy], color=c, lw=2, solid_capstyle="round")
        ax.plot(m_, yy, "o", ms=8, color=c, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.text(1.02, yy, fmt(m_), color=INK, fontsize="small", va="center",
                transform=ax.get_yaxis_transform())
    ax.axvline(ref, color=MUTED, lw=1, ls=(0, (3, 3)))
    ax.set_yticks(ypos, [SHORT.get(n, n) for n in names])
    ax.tick_params(length=0); ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.set_ylim(-0.6, len(names) - 0.3)


def fig_auc(results, wf, folder):
    """Ranking skill (AUC) on the fixed test period and in the walk-forward test."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), sharex=True)
    r = results[~results["model"].str.startswith("Benchmark: always")]
    _dot_whisker(axes[0], r["model"].tolist(), r["test_auc"].to_numpy(), r["test_auc_lo"].to_numpy(),
                 r["test_auc_hi"].to_numpy(), 0.5, lambda v: f"{v:.2f}")
    axes[0].set_title("Test period 2023-26 (54 calls)")
    w = wf[~wf["model"].str.startswith("Benchmark: always")]
    _dot_whisker(axes[1], w["model"].tolist(), w["auc"].to_numpy(), w["auc_lo"].to_numpy(),
                 w["auc_hi"].to_numpy(), 0.5, lambda v: f"{v:.2f}")
    axes[1].set_title(f"Walk-forward 2015-26 ({int(w['n_calls'].iloc[0])} calls)")
    axes[1].set_yticklabels([])
    fig.subplots_adjust(wspace=0.22)
    for ax in axes:
        ax.set_xlim(0.15, 0.75); ax.set_xlabel("AUC, with 90% interval (0.50 = no skill)")
    _save(fig, folder, "fig2_auc.png")


def fig_spread(wf, folder):
    """What trading on each model would have earned per call, walk-forward."""
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    _dot_whisker(ax, wf["model"].tolist(), wf["spread"].to_numpy() * 100, wf["spread_lo"].to_numpy() * 100,
                 wf["spread_hi"].to_numpy() * 100, 0.0, lambda v: f"{v:+.1f}")
    ax.set_xlabel("Long minus short, days 2 to 20, percentage points per call (90% intervals)")
    ax.set_title(f"Walk-forward backtest, 2015 to 2026 ({int(wf['n_calls'].iloc[0])} calls)")
    ax.set_xlim(-10, 6)
    _save(fig, folder, "fig3_backtest.png")


def fig_regime(signs, folder):
    """Rank correlation of each input with the outcome, before and after 2023."""
    nice = {"rel_fwd_1d": "Day-one move vs SOX", "sox_fwd_1d": "SOX move on day one",
            "ret_pre_20d": "Run-up into the call", "gross_margin": "Gross margin",
            "op_margin": "Operating margin", "rev_growth_yoy": "Revenue growth, year on year",
            "inventory_days": "Inventory days", "capex_to_sales": "Capex / sales",
            "gross_margin_chg": "Change in gross margin", "inventory_days_chg": "Change in inventory days",
            "tone_net": "Call tone", "tone_chg": "Change in call tone"}
    s = signs.sort_values("rho_2010_2022")
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ypos = np.arange(len(s))
    for yy, a, b in zip(ypos, s["rho_2010_2022"], s["rho_2023_2026"]):
        ax.plot([a, b], [yy, yy], color=GRID, lw=2, zorder=1)
    ax.scatter(s["rho_2010_2022"], ypos, s=64, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=2,
               label="2010 to 2022 (180 calls)")
    ax.scatter(s["rho_2023_2026"], ypos, s=64, color=ORANGE, edgecolor=SURFACE, linewidth=2, zorder=2,
               label="2023 to 2026 (54 calls)")
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(ypos, [nice[f] for f in s["feature"]])
    ax.set_xlabel("Rank correlation with the days 2 to 20 return vs SOX")
    ax.set_xlim(-0.3, 0.45); ax.tick_params(length=0); ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2,
              fontsize="small", labelcolor=INK)
    ax.set_title("The relationships did not hold across periods")
    _save(fig, folder, "fig4_regime.png")


def make_figures(df, results, wf, signs, folder):
    fig_outcome(df, folder)
    fig_auc(results, wf, folder)
    fig_spread(wf, folder)
    fig_regime(signs, folder)
