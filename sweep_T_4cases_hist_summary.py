# -*- coding: utf-8 -*-
"""
Summary plot for hist_T_4cases_data.npz.

Compresses each per-T histogram into a single number and plots three quantities
vs T (4 cases overlaid):

  (1) E[a_{1,l}^2]       -- should follow a^2 / T exactly  (log-log, slope -1)
  (2) E[p_l]             -- mean allele frequency
  (3) E[p_l (1 - p_l)]   -- directly drives Vg = 2 * sum_l ||a||^2 * p(1-p)

Output:
  hist_T_4cases_summary.pdf
"""

import os
import itertools
import numpy as np
import matplotlib.pyplot as plt

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

cases  = ['A', 'B', 'C', 'D']
colors = {'A': 'C0', 'B': 'C3', 'C': 'C2', 'D': 'C1'}
labels = {
    'A': r'A: $\Sigma_{ii}=\sigma^2,\ \Sigma_{ij}=+\sigma^2$',
    'B': r'B: $\Sigma_{ii}=\sigma^2,\ \Sigma_{ij}=-\sigma^2$',
    'C': r'C: $\Sigma_{ii}=\sigma^2/T,\ \Sigma_{ij}=+\sigma^2/T$',
    'D': r'D: $\Sigma_{ii}=\sigma^2/T,\ \Sigma_{ij}=-\sigma^2/T$',
}

for dist_name, dir_name, a2 in itertools.product(
        dist_names, dir_names, a2_values):
    tag = f"{dist_name}_{dir_name}_{A1_TAG}_a2_{a2:.2f}"
    DATA_FILE = f'hist_T_4cases_data_{tag}.npz'

    if not os.path.exists(DATA_FILE):
        print(f"[skip] {DATA_FILE} not found. Run sweep_T_4cases_violin.py first "
              "to generate the per-locus data for this (dist, a1, a2).")
        continue

    print(f"\n############## {DATA_FILE}  (dist = {dist_name}, dir = {dir_name}, "
          f"a2 = {a2:.3f}) ##############")
    d = np.load(DATA_FILE)
    T_list = d['T_list']

    mean_a1sq = {c: np.zeros(len(T_list)) for c in cases}
    sem_a1sq  = {c: np.zeros(len(T_list)) for c in cases}
    mean_p    = {c: np.zeros(len(T_list)) for c in cases}
    sem_p     = {c: np.zeros(len(T_list)) for c in cases}
    mean_pq   = {c: np.zeros(len(T_list)) for c in cases}
    sem_pq    = {c: np.zeros(len(T_list)) for c in cases}

    for ti, T in enumerate(T_list):
        for c in cases:
            a1sq = d[f'{c}_T{T}_a1sq']            # (L, rep)
            p    = d[f'{c}_T{T}_p']               # (L, rep)
            pq   = p * (1 - p)
            n    = a1sq.size
            mean_a1sq[c][ti] = a1sq.mean()
            sem_a1sq[c][ti]  = a1sq.std()  / np.sqrt(n)
            mean_p[c][ti]    = p.mean()
            sem_p[c][ti]     = p.std()     / np.sqrt(n)
            mean_pq[c][ti]   = pq.mean()
            sem_pq[c][ti]    = pq.std()    / np.sqrt(n)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # (1) E[a_{1,l}^2]: log-log, with a^2/T reference
    ax = axes[0]
    for c in cases:
        ax.errorbar(T_list, mean_a1sq[c], yerr=sem_a1sq[c],
                    marker='o', color=colors[c], label=labels[c], capsize=3)
    ax.plot(T_list, a2 / np.array(T_list), 'k--', lw=1,
            label=r'$a^2/T$ (theory)')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$T$')
    ax.set_ylabel(r'$E[a_{1,l}^2]$')
    ax.set_title(rf'(1)  $E[a_{{1,l}}^2]$  vs $T$   ($a^2={a2:.2f}$)'
                 '\n(slope $-1$, no case dependence)')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=7, loc='lower left')

    # (2) E[p_l]
    ax = axes[1]
    for c in cases:
        ax.errorbar(T_list, mean_p[c], yerr=sem_p[c],
                    marker='o', color=colors[c], label=labels[c], capsize=3)
    ax.set_xscale('log')
    ax.set_xlabel(r'$T$')
    ax.set_ylabel(r'$E[p_l]$')
    ax.set_title(rf'(2)  $E[p_l]$  vs $T$   ($a^2={a2:.2f}$)')
    ax.grid(True, which='both', alpha=0.3)

    # (3) E[p_l (1 - p_l)]  (drives Vg)
    ax = axes[2]
    for c in cases:
        ax.errorbar(T_list, mean_pq[c], yerr=sem_pq[c],
                    marker='o', color=colors[c], label=labels[c], capsize=3)
    ax.set_xscale('log')
    ax.set_xlabel(r'$T$')
    ax.set_ylabel(r'$E[p_l (1-p_l)]$')
    ax.set_title(rf'(3)  $E[p_l (1-p_l)]$  vs $T$   ($a^2={a2:.2f}$)'
                 '\n(direct driver of $V_g$)')
    ax.grid(True, which='both', alpha=0.3)

    fig.suptitle(f'A~{dist_name}, dir={dir_name}, $a^2$={a2:.2f}',
                 fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fname = f'hist_T_4cases_summary_{tag}.pdf'
    plt.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {fname}")

    # Also print the numbers
    print(f"\n=== E[a_{{1,l}}^2]  (a2={a2:.3f}) ===")
    print(f"{'T':>6} | " + "  ".join(f"{c:>10}" for c in cases) + "  |  a^2/T")
    for ti, T in enumerate(T_list):
        row = "  ".join(f"{mean_a1sq[c][ti]:10.3e}" for c in cases)
        print(f"{T:>6} | {row}  |  {a2/T:.3e}")

    print(f"\n=== E[p_l]  (a2={a2:.3f}) ===")
    print(f"{'T':>6} | " + "  ".join(f"{c:>10}" for c in cases))
    for ti, T in enumerate(T_list):
        row = "  ".join(f"{mean_p[c][ti]:10.4f}" for c in cases)
        print(f"{T:>6} | {row}")

    print(f"\n=== E[p_l (1 - p_l)]  (a2={a2:.3f}) ===")
    print(f"{'T':>6} | " + "  ".join(f"{c:>10}" for c in cases))
    for ti, T in enumerate(T_list):
        row = "  ".join(f"{mean_pq[c][ti]:10.4e}" for c in cases)
        print(f"{T:>6} | {row}")
