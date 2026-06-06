# -*- coding: utf-8 -*-
"""
Per-locus diagnostic histograms for the 4 covariance cases.

This script NO LONGER runs its own simulation.  It reads the per-locus data
produced by sweep_T_4cases_violin.py (hist_T_4cases_data_<tag>.npz), which
stores, for every (locus, replicate):
    a1_sq = a_{1,l}^2   (focal-trait squared effect)
    p     = p_l         (allele frequency)
and, for each T, draws a 4 x 2 grid of histograms (rows = cases A-D,
columns = [a_{1,l}^2, p_l]) pooled over all L loci and all replicates.

Run order:
    python sweep_T_4cases_violin.py    # simulate + save .npz + violin plots
    python sweep_T_4cases_hist.py      # this script: histograms from the .npz

Output:
    hist_T{T}_4cases_<tag>.pdf         -- one PDF per (a2, T)
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# a2 values swept: 0.01, 0.03, 0.1
a2_values = np.array([0.01, 0.03, 0.1])

colors = {'A': 'C0', 'B': 'C3', 'C': 'C2', 'D': 'C1'}
case_titles = {
    'A': r'A: $\Sigma_{ii}=\sigma^2,\ \Sigma_{ij}=+\sigma^2$',
    'B': r'B: $\Sigma_{ii}=\sigma^2,\ \Sigma_{ij}=-\sigma^2$',
    'C': r'C: $\Sigma_{ii}=\sigma^2/T,\ \Sigma_{ij}=+\sigma^2/T$',
    'D': r'D: $\Sigma_{ii}=\sigma^2/T,\ \Sigma_{ij}=-\sigma^2/T$',
}

for a2 in a2_values:
    tag = f"a2_{a2:.2f}"
    DATA_FILE = f'hist_T_4cases_data_{tag}.npz'

    if not os.path.exists(DATA_FILE):
        print(f"[skip] {DATA_FILE} not found. Run sweep_T_4cases_violin.py first "
              "to generate the per-locus data for this a2.")
        continue

    print(f"\n############## {DATA_FILE}  (a2 = {a2:.3f}) ##############")
    d = np.load(DATA_FILE)
    T_list = [int(T) for T in d['T_list']]

    for T in T_list:
        fig, axes = plt.subplots(4, 2, figsize=(10, 11), sharex='col')
        for ri, label in enumerate(['A', 'B', 'C', 'D']):
            a1_sq = d[f'{label}_T{T}_a1sq']
            p     = d[f'{label}_T{T}_p']
            L, rep = a1_sq.shape
            a1_flat = a1_sq.ravel()
            p_flat  = p.ravel()

            ax_a = axes[ri, 0]
            ax_a.hist(a1_flat, bins=60, color=colors[label], alpha=0.75)
            ax_a.set_ylabel(case_titles[label], fontsize=9)
            ax_a.set_yscale('log')
            ax_a.grid(True, alpha=0.3)
            ax_a.axvline(a2 / T, color='k', ls='--', lw=0.8,
                         label=f'a²/T = {a2/T:.2e}')
            ax_a.legend(fontsize=7, loc='upper right')

            ax_p = axes[ri, 1]
            ax_p.hist(p_flat, bins=60, range=(0, 1), color=colors[label], alpha=0.75)
            ax_p.set_yscale('log')
            ax_p.grid(True, alpha=0.3)

        axes[0, 0].set_title(r'$a_{1,l}^2$  (first-dim squared effect)', fontsize=11)
        axes[0, 1].set_title(r'$p_l$  (allele frequency)', fontsize=11)
        axes[-1, 0].set_xlabel(r'$a_{1,l}^2$')
        axes[-1, 1].set_xlabel(r'$p_l$')

        fig.suptitle(f'Per-locus distributions, T = {T}, $a^2$ = {a2:.2f}  '
                     f'(L={L}, rep={rep}, pooled over all loci × reps)',
                     fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        fname = f'hist_T{T}_4cases_{tag}.pdf'
        plt.savefig(fname, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {fname}")
