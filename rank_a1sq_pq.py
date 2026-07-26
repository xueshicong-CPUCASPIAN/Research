# -*- coding: utf-8 -*-
"""
Per-locus RANK plots for the 4 covariance cases A, B, C, D and each T in T_list.

Three quantities are plotted, each in its own figure (one panel per T):

    1. c_l = a_{1,l}^2 * p_l * (1 - p_l)   -- per-locus contribution to V_g(trait 1)
    2.       a_{1,l}^2                      -- focal-trait squared effect (segregating loci)
    3.       p_l                            -- mutant allele frequency

For each replicate the L loci are sorted in DECREASING order of the quantity,
giving a length-L curve (rank 1 = largest).  At each rank we summarise across
replicates with BOTH the MEDIAN (solid) and the MEAN (dashed), plus a 25-75%
band.  The median sits in the middle of its band; the mean rides above it because
the across-replicate distribution is right-skewed.  For quantity 1 the mean is the
V_g-additive summary (E[V_g]=2*sum_l mean_l).

TAIL CUT-OFF.  Most of the L loci are monomorphic (p=0), so the sorted curves
collapse to zero in a long tail that carries no information and drops suddenly on
the log axis.  We therefore show only the informative HEAD: the x-axis is cut at
`TOP_RANKS` (auto-detected from the data if left as None), which also lets the
y-axis autoscale to the head so the meaningful loci are large and clear.  For
quantities 2 and 3 the non-segregating loci (p=0) are sent to zero so they fall in
the cut-off tail; quantity 2 thus shows effect sizes among segregating loci only.

The static-optimum (sigma^2 = 0) baseline is overlaid in black so the fluctuating
cases A-D can be compared against it rank-by-rank.

Reuses the per-locus data produced by sweep_T_4cases_violin.py
(hist_T_4cases_data_<tag>.npz for A-D, hist_baseline_sigma0_<a2>.npz for the
sigma^2 = 0 baseline), so no simulation is run here.

Output (one set per (dist, a1, a2)):
    rank_a1sq_pq_4cases_<tag>.pdf   -- quantity 1 (a_1^2 p(1-p))
    rank_a1sq_4cases_<tag>.pdf      -- quantity 2 (a_1^2)
    rank_p_4cases_<tag>.pdf         -- quantity 3 (p)
"""

import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── tail cut-off controls ──────────────────────────────────────────────────────
# TOP_RANKS: fixed number of leading ranks to show (int), or None to auto-detect.
# Auto rule: keep ranks whose across-replicate MEAN exceeds TAIL_REL * (rank-1 mean);
# the cut-off is the largest such rank over all shown series (cases + baseline),
# clamped to [RANK_MIN, L] with a small margin.  Lower TAIL_REL -> longer tail kept.
TOP_RANKS = None      # e.g. set to 30 to force a fixed window
TAIL_REL  = 1e-2      # auto cut-off threshold, relative to the rank-1 mean
RANK_MIN  = 10        # never show fewer leading ranks than this
RANK_PAD  = 2         # extra ranks kept past the detected shoulder

# a2 (mean effect size) swept; currently a single value
a2_values = np.array([0.03])
# A-scale distributions produced by sweep_T_4cases_violin.py (must match its `dists` keys)
dist_names = ['const']  # complex A-scale dists disabled: 'twopoint', 'exp', 'gamma', 'lognormal'
# per-trait scaling a_t ~ N(0, A / T**p) (must match violin's `a1_scalings`); name -> p
a1_scalings = {'aT1': 1.0, 'aTsqrt': 0.5}

cases = ['A', 'B', 'C', 'D']
colors = {'A': 'C0', 'B': 'C3', 'C': 'C2', 'D': 'C1'}
case_titles = {
    'A': r'A: $\Sigma_{ii}=\sigma^2,\ \Sigma_{ij}=+\sigma^2$',
    'B': r'B: $\Sigma_{ii}=\sigma^2,\ \Sigma_{ij}=-\sigma^2$',
    'C': r'C: $\Sigma_{ii}=\sigma^2/T,\ \Sigma_{ij}=+\sigma^2/T$',
    'D': r'D: $\Sigma_{ii}=\sigma^2/T,\ \Sigma_{ij}=-\sigma^2/T$',
}

# ── the three quantities to rank (name -> (per-locus map, y-label, file stem, title)) ──
# Each map takes the per-locus (a1_sq, p) arrays (L, rep) and returns the quantity
# to be sorted descending.  Non-segregating loci (p=0) are sent to 0 in all three so
# they land in the cut-off tail rather than polluting the head.
quantities = {
    'a1sq_pq': (lambda a1_sq, p: a1_sq * p * (1 - p),
                r'$a_{1,l}^2\, p_l (1-p_l)$', 'rank_a1sq_pq',
                r'$a_{1,l}^2\, p_l(1-p_l)$'),
    'a1sq':    (lambda a1_sq, p: a1_sq * (p > 0),
                r'$a_{1,l}^2$', 'rank_a1sq',
                r'$a_{1,l}^2$ (segregating loci)'),
    'p':       (lambda a1_sq, p: p,
                r'$p_l$', 'rank_p',
                r'$p_l$'),
}


def rank_summary(c):
    """Sort per-locus quantity c=(L,rep) descending per replicate; return ranks and,
    across replicates at each rank: median, mean, and the 25/75% band."""
    c_sorted = -np.sort(-c, axis=0)             # (L, rep), rank 0 = largest
    ranks = np.arange(1, c_sorted.shape[0] + 1)
    med  = np.quantile(c_sorted, 0.50, axis=1)  # median -> middle of the band
    mean = c_sorted.mean(axis=1)                # mean -> V_g-additive, above median
    lo   = np.quantile(c_sorted, 0.25, axis=1)
    hi   = np.quantile(c_sorted, 0.75, axis=1)
    return ranks, med, mean, lo, hi


def head_cutoff(mean_curves):
    """Largest informative rank across a list of mean arrays (1 = rank-1).

    Rank r is informative while its mean exceeds TAIL_REL * (largest rank-1 mean).
    Returns an int in [RANK_MIN, L]; honoured only when TOP_RANKS is None."""
    L = len(mean_curves[0])
    if TOP_RANKS is not None:
        return int(np.clip(TOP_RANKS, 1, L))
    ref = max(m[0] for m in mean_curves)
    if ref <= 0:
        return L
    last = RANK_MIN
    for m in mean_curves:
        idx = np.where(m >= TAIL_REL * ref)[0]
        if idx.size:
            last = max(last, idx[-1] + 1)       # +1: index -> 1-based rank
    return int(np.clip(last + RANK_PAD, RANK_MIN, L))


for dist_name, (a1_name, texp), a2 in itertools.product(
        dist_names, a1_scalings.items(), a2_values):
    tag = f"{dist_name}_{a1_name}_a2_{a2:.2f}"
    DATA_FILE = f'hist_T_4cases_data_{tag}.npz'

    if not os.path.exists(DATA_FILE):
        print(f"[skip] {DATA_FILE} not found. Run sweep_T_4cases_violin.py first "
              "to generate the per-locus data for this (dist, a1, a2).")
        continue

    print(f"\n############## {DATA_FILE}  (dist = {dist_name}, a1 = {a1_name}, a2 = {a2:.3f}) ##############")
    npz = np.load(DATA_FILE)
    T_list = list(npz['T_list'])

    # σ²=0 static baseline (same a1-scaling, cases coincide) for overlay, if present
    BASE_FILE = f'hist_baseline_sigma0_a2_{a2:.2f}.npz'
    base_npz = np.load(BASE_FILE) if os.path.exists(BASE_FILE) else None
    if base_npz is None:
        print(f"[note] {BASE_FILE} not found; σ²=0 baseline not overlaid.")

    # ── one figure per (quantity); one panel per T ─────────────────────────────
    for qname, (qmap, ylabel, stem, qtitle) in quantities.items():
        n = len(T_list)
        ncol = 2
        nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(6.5 * ncol, 4.2 * nrow),
                                 squeeze=False)
        axes_flat = axes.ravel()

        for pi, T in enumerate(T_list):
            ax = axes_flat[pi]

            # summarise every series first so the tail cut-off can be shared per panel
            series = {}   # label -> (ranks, med, mean, lo, hi)
            for label in cases:
                series[label] = rank_summary(
                    qmap(npz[f'{label}_T{T}_a1sq'], npz[f'{label}_T{T}_p']))
            if base_npz is not None:
                series['baseline'] = rank_summary(
                    qmap(base_npz[f'{a1_name}_T{T}_a1sq'],
                         base_npz[f'{a1_name}_T{T}_p']))

            cut = head_cutoff([s[2] for s in series.values()])   # s[2] = mean

            # colour = case (black = σ²=0 baseline); solid = median, dashed = mean.
            for label in cases:
                ranks, med, mean, lo, hi = series[label]
                sl = slice(0, cut)
                ax.plot(ranks[sl], med[sl],  color=colors[label], lw=1.6, ls='-',
                        label=case_titles[label])          # median (only this is labelled)
                ax.plot(ranks[sl], mean[sl], color=colors[label], lw=1.1, ls='--')  # mean
                ax.fill_between(ranks[sl], lo[sl], hi[sl], color=colors[label], alpha=0.15)

            # σ²=0 static baseline overlay (cases coincide -> single black curve)
            if 'baseline' in series:
                ranks, med, mean, lo, hi = series['baseline']
                sl = slice(0, cut)
                ax.plot(ranks[sl], med[sl],  color='k', lw=1.8, ls='-',
                        label=r'$\sigma^2=0$ baseline')
                ax.plot(ranks[sl], mean[sl], color='k', lw=1.1, ls='--')
                ax.fill_between(ranks[sl], lo[sl], hi[sl], color='k', alpha=0.12)

            ax.set_yscale('log')
            ax.set_xlim(1, cut)
            ax.set_title(f'T = {T}', fontsize=11)
            ax.set_xlabel(f'Locus rank (1 = largest; top {cut} shown)')
            ax.set_ylabel(ylabel)
            ax.grid(True, which='both', alpha=0.3)
            if pi == 0:
                # case/baseline entries + a style key explaining solid vs dashed
                handles, labs = ax.get_legend_handles_labels()
                style_key = [Line2D([0], [0], color='gray', ls='-',  label='median'),
                             Line2D([0], [0], color='gray', ls='--', label='mean')]
                ax.legend(handles + style_key, labs + ['median', 'mean'],
                          fontsize=8, loc='upper right')

        # hide any unused panels
        for pi in range(n, len(axes_flat)):
            axes_flat[pi].axis('off')

        fig.suptitle(
            rf'Rank plot of per-locus {qtitle}  '
            rf'(A~{dist_name}, $a_t$~{a1_name}, $a^2$={a2:.2f})'
            '\n(solid = median, dashed = mean, band = 25–75%; '
            r'colour = case, black = $\sigma^2{=}0$ baseline; tail cut, head zoomed)',
            fontsize=12,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        fname = f'{stem}_4cases_{tag}.pdf'
        plt.savefig(fname, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {fname}")
