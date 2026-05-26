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
import numpy as np
import matplotlib.pyplot as plt

DATA_FILE = 'hist_T_4cases_data_a_0.01.npz'

if not os.path.exists(DATA_FILE):
    raise SystemExit(
        f"{DATA_FILE} not found. Run sweep_T_4cases_hist.py first to "
        "generate the per-locus data."
    )

npz = np.load(DATA_FILE)
T_list = list(npz['T_list'])

cases = ['A', 'B', 'C', 'D']
colors = {'A': 'C0', 'B': 'C3', 'C': 'C2', 'D': 'C1'}
case_titles = {
    'A': r'A: $\Sigma_{ii}=\sigma^2,\ \Sigma_{ij}=+\sigma^2$',
    'B': r'B: $\Sigma_{ii}=\sigma^2,\ \Sigma_{ij}=-\sigma^2$',
    'C': r'C: $\Sigma_{ii}=\sigma^2/T,\ \Sigma_{ij}=+\sigma^2/T$',
    'D': r'D: $\Sigma_{ii}=\sigma^2/T,\ \Sigma_{ij}=-\sigma^2/T$',
}

# ── one figure per T ─────────────────────────────────────────────────────────
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
        rf'Per-locus $a_{{1,l}}^2\, p_l(1-p_l)$,  T = {T}'
        '\n(pooled over all loci × replicates)',
        fontsize=12,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fname = f'hist_a1sq_pq_T{T}_4cases_a_0.01.pdf'
    plt.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {fname}")
