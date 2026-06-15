# -*- coding: utf-8 -*-
"""
Summary plot for the per-locus quantity   a_{1,l}^2 * p_l * (1 - p_l).

Uses the per-a2 data files produced by sweep_T_4cases_hist.py
(`hist_T_4cases_data_a2_<value>.npz`) and compresses each per-T histogram
(as drawn by hist_a1sq_pq.py -> hist_a1sq_pq_T{T}_4cases_a2_<value>.pdf)
into summary statistics plotted against T for the 4 cases:

  (1) E[a_{1,l}^2 * p_l (1-p_l)]      -- per-locus contribution to V_g(trait 1)
  (2) sum_l a_{1,l}^2 * p_l (1-p_l)   -- per-replicate total (V_g, trait 1, /2)

Sweeps a2 over [0.01, 0.03, 0.1].

Output (one per a2 value):
  hist_a1sq_pq_summary_a2_<value>.pdf
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

cases  = ['A', 'B', 'C', 'D']
colors = {'A': 'C0', 'B': 'C3', 'C': 'C2', 'D': 'C1'}
labels = {
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
    d = np.load(DATA_FILE)
    T_list = d['T_list']

    mean_per_locus = {c: np.zeros(len(T_list)) for c in cases}
    sem_per_locus  = {c: np.zeros(len(T_list)) for c in cases}
    mean_sum_rep   = {c: np.zeros(len(T_list)) for c in cases}
    sem_sum_rep    = {c: np.zeros(len(T_list)) for c in cases}

    for ti, T in enumerate(T_list):
        for c in cases:
            a1sq = d[f'{c}_T{T}_a1sq']               # (L, rep)
            p    = d[f'{c}_T{T}_p']                  # (L, rep)
            v    = a1sq * p * (1 - p)                # (L, rep)

            # (1) per-locus mean across all loci × replicates
            flat = v.ravel()
            n = flat.size
            mean_per_locus[c][ti] = flat.mean()
            sem_per_locus[c][ti]  = flat.std(ddof=1) / np.sqrt(n)

            # (2) per-replicate total: sum over loci, then mean/SEM across replicates
            rep_total = v.sum(axis=0)                # (rep,)
            nr = rep_total.size
            mean_sum_rep[c][ti] = rep_total.mean()
            sem_sum_rep[c][ti]  = rep_total.std(ddof=1) / np.sqrt(nr)

    # ── plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # (1) per-locus mean vs T
    ax = axes[0]
    for c in cases:
        ax.errorbar(T_list, mean_per_locus[c], yerr=sem_per_locus[c],
                    marker='o', color=colors[c], label=labels[c], capsize=3)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$T$')
    ax.set_ylabel(r'$E\!\left[a_{1,l}^2\, p_l (1-p_l)\right]$')
    ax.set_title(rf'(1)  per-locus $E[a_{{1,l}}^2\, p_l(1-p_l)]$  vs $T$   '
                 rf'($a^2={a2:.2f}$, $a_t$~{a1_name})')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=7, loc='best')

    # (2) per-replicate total vs T (this is V_g(trait 1) / 2)
    ax = axes[1]
    for c in cases:
        ax.errorbar(T_list, mean_sum_rep[c], yerr=sem_sum_rep[c],
                    marker='o', color=colors[c], label=labels[c], capsize=3)
    ax.set_xscale('log')
    ax.set_xlabel(r'$T$')
    ax.set_ylabel(r'$\sum_l a_{1,l}^2\, p_l (1-p_l)$')
    ax.set_title(rf'(2)  per-replicate $\sum_l a_{{1,l}}^2 p_l(1-p_l)$  vs $T$   '
                 rf'($a^2={a2:.2f}$, $a_t$~{a1_name})' '\n($V_g$(trait 1)$/2$)')
    ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    fname = f'hist_a1sq_pq_summary_{tag}.pdf'
    plt.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {fname}")

    # ── print numbers ────────────────────────────────────────────────────────
    print(f"\n=== E[a_{{1,l}}^2 * p_l (1-p_l)]  (per locus, a2={a2:.3f}) ===")
    print(f"{'T':>6} | " + "  ".join(f"{c:>11}" for c in cases))
    for ti, T in enumerate(T_list):
        row = "  ".join(f"{mean_per_locus[c][ti]:11.3e}" for c in cases)
        print(f"{T:>6} | {row}")

    print(f"\n=== sum_l a_{{1,l}}^2 * p_l (1-p_l)  (per replicate, mean, a2={a2:.3f}) ===")
    print(f"{'T':>6} | " + "  ".join(f"{c:>11}" for c in cases))
    for ti, T in enumerate(T_list):
        row = "  ".join(f"{mean_sum_rep[c][ti]:11.3e}" for c in cases)
        print(f"{T:>6} | {row}")
