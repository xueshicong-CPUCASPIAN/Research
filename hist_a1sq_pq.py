# -*- coding: utf-8 -*-
"""
Per-locus histogram of the quantity   a_{1,l}^2 * p_l * (1 - p_l)
for the 4 covariance cases A, B, C, D and each T in T_list.

This is the per-locus contribution to the additive genetic variance for
trait 1 (up to a factor of 2). It reuses the data produced by
`sweep_T_4cases_hist.py` (file `hist_T_4cases_data.npz`); if that file is
absent it re-runs the simulation.

Output:
    hist_a1sq_pq_T{T}_4cases.pdf  -- one PDF per T value (4 cases stacked)
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

    # ── one figure per T ─────────────────────────────────────────────────────
    for T in T_list:
        fig, axes = plt.subplots(4, 1, figsize=(7, 11), sharex=True)

        # Pre-compute global x-range across the 4 cases for this T
        all_vals = []
        case_vals = {}
        for label in cases:
            a1_sq = npz[f'{label}_T{T}_a1sq']
            p     = npz[f'{label}_T{T}_p']
            v = (a1_sq * p * (1 - p)).ravel()
            case_vals[label] = v
            all_vals.append(v)
        all_vals = np.concatenate(all_vals)
        poly_mask = all_vals > 0
        if poly_mask.any():
            xmax = np.quantile(all_vals[poly_mask], 0.999)
        else:
            xmax = all_vals.max() if all_vals.size else 1.0
        xmax = max(xmax, 1e-12)

        for ri, label in enumerate(cases):
            v = case_vals[label]
            ax = axes[ri]
            ax.hist(v, bins=80, range=(0, xmax), color=colors[label], alpha=0.8)
            ax.set_yscale('log')
            ax.set_ylabel(case_titles[label], fontsize=9)
            ax.grid(True, alpha=0.3)

            mean_v = v.mean()
            poly_frac = np.mean(v > 0)
            ax.axvline(mean_v, color='k', ls='--', lw=0.8,
                       label=f'mean = {mean_v:.2e}\npoly frac = {poly_frac:.3f}')
            ax.legend(fontsize=7, loc='upper right')

        axes[-1].set_xlabel(r'$a_{1,l}^2\, p_l (1 - p_l)$')
        fig.suptitle(
            rf'Per-locus $a_{{1,l}}^2\, p_l(1-p_l)$,  T = {T}, $a^2$ = {a2:.2f}, $a_t$~{a1_name}'
            '\n(pooled over all loci × replicates)',
            fontsize=12,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fname = f'hist_a1sq_pq_T{T}_4cases_{tag}.pdf'
        plt.savefig(fname, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {fname}")
