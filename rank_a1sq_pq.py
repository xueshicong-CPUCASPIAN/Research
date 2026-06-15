# -*- coding: utf-8 -*-
"""
Per-locus RANK plot of the quantity   c_l = a_{1,l}^2 * p_l * (1 - p_l)
for the 4 covariance cases A, B, C, D and each T in T_list.

c_l is the per-locus contribution to the additive genetic variance of trait 1
(V_g(trait 1) = 2 * sum_l c_l).  For each replicate the L loci are sorted in
DECREASING order, giving a length-L curve (rank 1 = largest contributor); these
curves are then averaged across replicates at each rank.  The result shows the
SHAPE of the decay -- whether a few loci dominate V_g (steep) or the contribution
is spread evenly across loci (shallow).  A shaded band shows the 25-75% spread
across replicates at each rank.

Reuses the per-locus data produced by sweep_T_4cases_violin.py
(hist_T_4cases_data_<tag>.npz), so no simulation is run here.

Output (one per (dist, a1, a2)):
    rank_a1sq_pq_4cases_<tag>.pdf   -- one panel per T, 4 cases overlaid, log-y
"""

import os
import itertools
import numpy as np
import matplotlib.pyplot as plt

# a2 (mean effect size) swept; currently a single value
a2_values = np.array([0.03])
# A-scale distributions produced by sweep_T_4cases_violin.py (must match its `dists` keys)
dist_names = ['twopoint', 'exp', 'const', 'gamma', 'lognormal']
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

    # ── one figure per (dist, a1, a2); one panel per T ─────────────────────────
    n = len(T_list)
    ncol = 2
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.5 * ncol, 4.2 * nrow),
                             squeeze=False)
    axes_flat = axes.ravel()

    for pi, T in enumerate(T_list):
        ax = axes_flat[pi]
        for label in cases:
            a1_sq = npz[f'{label}_T{T}_a1sq']         # (L, rep)
            p     = npz[f'{label}_T{T}_p']            # (L, rep)
            c = a1_sq * p * (1 - p)                   # per-locus contribution (L, rep)

            # sort DESCENDING within each replicate, then summarise across reps per rank
            c_sorted = -np.sort(-c, axis=0)           # (L, rep), rank 0 = largest
            L = c_sorted.shape[0]
            ranks = np.arange(1, L + 1)
            mean_curve = c_sorted.mean(axis=1)        # mean contribution at each rank
            lo = np.quantile(c_sorted, 0.25, axis=1)
            hi = np.quantile(c_sorted, 0.75, axis=1)

            ax.plot(ranks, mean_curve, color=colors[label], lw=1.6,
                    label=case_titles[label])
            ax.fill_between(ranks, lo, hi, color=colors[label], alpha=0.15)

        ax.set_yscale('log')
        ax.set_title(f'T = {T}', fontsize=11)
        ax.set_xlabel('Locus rank (1 = largest)')
        ax.set_ylabel(r'$a_{1,l}^2\, p_l (1-p_l)$')
        ax.grid(True, which='both', alpha=0.3)
        if pi == 0:
            ax.legend(fontsize=8, loc='upper right')

    # hide any unused panels
    for pi in range(n, len(axes_flat)):
        axes_flat[pi].axis('off')

    fig.suptitle(
        rf'Rank plot of per-locus $a_{{1,l}}^2\, p_l(1-p_l)$  '
        rf'(A~{dist_name}, $a_t$~{a1_name}, $a^2$={a2:.2f})'
        '\n(sorted per replicate, mean across replicates; band = 25–75%)',
        fontsize=12,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fname = f'rank_a1sq_pq_4cases_{tag}.pdf'
    plt.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {fname}")
