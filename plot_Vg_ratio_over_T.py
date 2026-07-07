# -*- coding: utf-8 -*-
"""
Plot the ratio  V_g(T) / V_g|_{sigma^2 = 0}  against the number of trait
dimensions T, for the 4 covariance cases A, B, C, D.

Numerator   V_g(T)          : simulated genetic variance of the focal trait,
                              read from Vg_sweep_T_4cases_<tag>.npz produced by
                              sweep_T_4cases_violin.py (fluctuating optimum,
                              sigma^2 = 1e-3).
Denominator V_g|_{sigma^2=0}: the ANALYTIC static-optimum baseline, computed
                              straight from the parameters (no simulation):

    V_g|_{sigma^2=0} = 4 * L * mu * V_s        (Latter-Bulmer / house-of-cards)

  This is the equilibrium additive genetic variance of one trait under pure
  Gaussian stabilizing selection with a fixed optimum.  It is independent of the
  effect size a^2 and of T (each trait is selected independently with width V_s),
  so it is a single horizontal reference level.

A ratio of 1 (dashed horizontal line) therefore means the fluctuating optimum
adds no variance beyond the static-optimum baseline.
"""

import os
import itertools
import numpy as np
import matplotlib.pyplot as plt

# ── parameters (must match sweep_T_4cases_violin.py) ───────────────────────────
L         = 100
mu        = 6.6e-6
V_s       = 5

# analytic static-optimum baseline (parameter-only, no simulation)
Vg_static = 4 * L * mu * V_s
print(f"Analytic baseline  V_g|sigma^2=0 = 4*L*mu*V_s = {Vg_static:.6g}")

# tags to look for (must match the sweep's dists / a1_scalings / a2_values)
a2_values   = np.array([0.03])
dist_names  = ['const']                 # complex A-scale dists disabled
a1_scalings = {'aT1': 1.0, 'aTsqrt': 0.5}

cases   = ['A', 'B', 'C', 'D']
colors  = {'A': 'C0', 'B': 'C3', 'C': 'C2', 'D': 'C1'}
labels  = {
    'A': r'A: $\Sigma_{ii}=\sigma^2,\ \Sigma_{ij}=+\sigma^2$',
    'B': r'B: $\Sigma_{ii}=\sigma^2,\ \Sigma_{ij}=-\sigma^2$',
    'C': r'C: $\Sigma_{ii}=\sigma^2/T,\ \Sigma_{ij}=+\sigma^2/T$',
    'D': r'D: $\Sigma_{ii}=\sigma^2/T,\ \Sigma_{ij}=-\sigma^2/T$',
}

# ── one figure per (dist, a1, a2) ──────────────────────────────────────────────
for dist_name, (a1_name, _), a2 in itertools.product(
        dist_names, a1_scalings.items(), a2_values):
    tag       = f"{dist_name}_{a1_name}_a2_{a2:.2f}"
    DATA_FILE = f'Vg_sweep_T_4cases_{tag}.npz'
    BASE_FILE = f'Vg_baseline_sigma0_a2_{a2:.2f}.npz'

    if not os.path.exists(DATA_FILE):
        print(f"[skip] {DATA_FILE} not found. Run sweep_T_4cases_violin.py first.")
        continue

    npz    = np.load(DATA_FILE)
    T_list = npz['T_list']

    # Denominator: the *simulated* σ²=0 baseline (per T) if available, else the
    # analytic constant 4LμVs. The baseline file stores one array per a1-scaling.
    if os.path.exists(BASE_FILE):
        Vg_base = np.load(BASE_FILE)[a1_name].mean(axis=1)   # (len(T_list),)
        denom_label = r'simulated $V_g|_{\sigma^2=0}$'
    else:
        Vg_base = np.full(len(T_list), Vg_static)            # analytic fallback
        denom_label = r'analytic $4L\mu V_s$'
        print(f"[note] {BASE_FILE} not found; normalising by analytic {Vg_static:.4g}.")

    fig, ax = plt.subplots(figsize=[8, 6])
    for c in cases:
        Vg      = npz[c]                       # (len(T_list), rep)
        ratio   = Vg.mean(axis=1) / Vg_base    # mean over replicates, then normalise
        sem     = Vg.std(axis=1) / np.sqrt(Vg.shape[1]) / Vg_base
        ax.errorbar(T_list, ratio, yerr=sem, marker='o', capsize=3,
                    color=colors[c], label=labels[c])

    ax.axhline(1.0, color='k', ls='--', lw=1,
               label=f'baseline ({denom_label}) = 1')
    # analytic Latter–Bulmer level, shown relative to the (simulated) baseline
    ax.plot(T_list, Vg_static / Vg_base, 'k:', marker='s', ms=4,
            label=r'analytic $4L\mu V_s$ / baseline')
    ax.set_xscale('log')
    ax.set_xticks(T_list)
    ax.set_xticklabels([str(int(T)) for T in T_list])
    ax.set_xlabel('number of trait dimensions  T')
    ax.set_ylabel(rf'$V_g(T)\,/\,${denom_label}')
    ax.set_title(
        rf'Genetic variance relative to static-optimum baseline'
        '\n'
        rf'(analytic $4L\mu V_s={Vg_static:.4g}$;  '
        rf'A~{dist_name}, $a_t$~{a1_name}, $a^2$={a2:.2f})'
    )
    ax.legend(fontsize=8, loc='best')
    fig.tight_layout()

    out = f'Vg_ratio_over_T_{tag}.pdf'
    fig.savefig(out)
    print(f"Saved {out}")

plt.show()
