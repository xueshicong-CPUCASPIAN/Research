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
import itertools
import numpy as np
import matplotlib.pyplot as plt

# ── output directory ──────────────────────────────────────────────────────────
# Read the .npz files written by sweep_T_4cases_violin.py, and put the figures next
# to them.  Must match the RESULTS_DIR set in that script.
RESULTS_DIR = 'results Aug 10'
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', RESULTS_DIR)
def opath(name):  return os.path.join(OUTDIR, name)

# a2 (mean effect size) swept; currently a single value
a2_values = np.array([0.03])
# A-scale distributions produced by sweep_T_4cases_violin.py (must match its `dists` keys)
dist_names = ['const']  # complex A-scale dists disabled: 'twopoint', 'exp', 'gamma', 'lognormal'
# per-trait DIRECTION distributions (must match violin's `dir_dists` keys):
#   'gauss' -- a_t ~ N(0, A/T)   ; 'pm' -- a_t = +-sqrt(A/T), the paper's model
dir_names = ['gauss', 'pm']
# per-trait scaling is fixed at a_t ~ sqrt(A/T) * direction; A1_TAG must match
# violin's `A1_TAG` since it appears in the filenames and baseline .npz keys.
A1_TAG = 'aT1'

colors = {'A': 'C0', 'B': 'C3', 'C': 'C2', 'D': 'C1'}
case_titles = {
    'A': r'A: $\Sigma_{ii}=\sigma^2,\ \Sigma_{ij}=+\sigma^2$',
    'B': r'B: $\Sigma_{ii}=\sigma^2,\ \Sigma_{ij}=-\sigma^2$',
    'C': r'C: $\Sigma_{ii}=\sigma^2/T,\ \Sigma_{ij}=+\sigma^2/T$',
    'D': r'D: $\Sigma_{ii}=\sigma^2/T,\ \Sigma_{ij}=-\sigma^2/T$',
}

for dist_name, dir_name, a2 in itertools.product(
        dist_names, dir_names, a2_values):
    tag = f"{dist_name}_{dir_name}_{A1_TAG}_a2_{a2:.2f}"
    DATA_FILE = opath(f'hist_T_4cases_data_{tag}.npz')

    if not os.path.exists(DATA_FILE):
        print(f"[skip] {DATA_FILE} not found. Run sweep_T_4cases_violin.py first "
              "to generate the per-locus data for this (dist, a1, a2).")
        continue

    print(f"\n############## {DATA_FILE}  (dist = {dist_name}, dir = {dir_name}, "
          f"a2 = {a2:.3f}) ##############")
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

        fig.suptitle(f'Per-locus distributions, T = {T}, $a^2$ = {a2:.2f}, '
                     f'dir={dir_name}  '
                     f'(L={L}, rep={rep}, pooled over all loci × reps)',
                     fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        fname = f'hist_T{T}_4cases_{tag}.pdf'
        plt.savefig(opath(fname), bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {fname}")
