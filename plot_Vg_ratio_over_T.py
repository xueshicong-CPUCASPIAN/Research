# -*- coding: utf-8 -*-
"""
Plot V_g(T) / (4 L mu V_s / T) against the number of trait dimensions T, for the
4 covariance cases A, B, C, D.

The denominator is the ANALYTIC static-optimum baseline, computed straight from
the parameters (no simulation):

    4 * L * mu * V_s / T    (multi-trait house-of-cards / Latter-Bulmer)

  = the equilibrium additive genetic variance of the FOCAL trait under pure
  Gaussian stabilizing selection with a fixed optimum, when each mutation has a
  T-dimensional effect vector.  The sim reports focal-trait V_g = 2*sum a_1^2 pq
  (trait 0 only) while stabilizing selection acts on the full ||a||^2 across all
  T traits; house-of-cards then gives V_g,1 = 4 L mu V_s * E[a_1^2/||a||^2], and
  for i.i.d. trait components E[a_1^2/||a||^2] = 1/T exactly.  So the baseline is
  4 L mu V_s / T (independent of a^2, but 1/T in the number of traits).  Dividing
  by it puts the static prediction at the constant reference level 1 (dashed
  horizontal line) for every T.

Two families of curves are shown, both divided by that analytic baseline:
  • simulated V_g|_{sigma^2=0} / analytic  (black squares) -- how well the static
    simulation matches the multi-trait Latter-Bulmer theory at each T;
  • simulated V_g / analytic for cases A-D (fluctuating optimum, sigma^2 = 1e-3),
    read from Vg_sweep_T_4cases_<tag>.npz produced by sweep_T_4cases_violin.py.

So a colored curve above 1 means the fluctuating optimum inflates V_g above the
analytic static baseline; the black curve shows how far the *simulated* static
baseline itself sits from that analytic line.
"""

import os
import itertools
import numpy as np
import matplotlib.pyplot as plt

# ── parameters (must match sweep_T_4cases_violin.py) ───────────────────────────
L         = 100
mu        = 6.6e-6
V_s       = 5

# analytic static-optimum baseline (parameter-only, no simulation).
# Multi-trait house-of-cards: focal-trait V_g|sigma^2=0 = 4*L*mu*V_s / T, because
# stabilizing selection acts on ||a||^2 over all T traits while only trait 0's
# variance is measured, and E[a_1^2/||a||^2] = 1/T for i.i.d. components.
# Kept as a per-T quantity: Vg_static(T) is looked up per T below.
Vg_static_1 = 4 * L * mu * V_s          # the T=1 value, = single-trait Latter-Bulmer
print(f"Analytic baseline  V_g|sigma^2=0 = 4*L*mu*V_s / T "
      f"(T=1 value = {Vg_static_1:.6g})")

# tags to look for (must match the sweep's dists / a1_scalings / a2_values)
a2_values   = np.array([0.03])
dist_names  = ['const']                 # complex A-scale dists disabled
# per-trait DIRECTION distributions (must match violin's `dir_dists` keys):
#   'gauss' -- a_t ~ N(0, A/T**p)   ; 'pm' -- a_t = +-sqrt(A/T**p), the paper's model
dir_names   = ['gauss', 'pm']
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
# Everything is normalised by the ANALYTIC baseline 4LμVs, so the dashed line at 1
# IS the analytic static-optimum prediction. Two things are shown against it:
#   • simulated σ²=0 baseline / analytic   -> how close the static sim is to theory
#   • simulated σ²>0 cases A–D / analytic  -> the fluctuating-optimum variance
for dist_name, dir_name, (a1_name, _), a2 in itertools.product(
        dist_names, dir_names, a1_scalings.items(), a2_values):
    tag       = f"{dist_name}_{dir_name}_{a1_name}_a2_{a2:.2f}"
    DATA_FILE = f'Vg_sweep_T_4cases_{tag}.npz'
    BASE_FILE = f'Vg_baseline_sigma0_{dir_name}_a2_{a2:.2f}.npz'

    if not os.path.exists(DATA_FILE):
        print(f"[skip] {DATA_FILE} not found. Run sweep_T_4cases_violin.py first.")
        continue

    npz    = np.load(DATA_FILE)
    T_list = npz['T_list']

    # per-T analytic baseline 4LμVs/T (matches the array order of T_list)
    Vg_static = Vg_static_1 / np.asarray(T_list, dtype=float)   # (len(T_list),)

    fig, ax = plt.subplots(figsize=[8, 6])

    # simulated σ²>0 cases, each divided by the per-T analytic baseline
    for c in cases:
        Vg    = npz[c]                          # (len(T_list), rep)
        ratio = Vg.mean(axis=1) / Vg_static
        sem   = Vg.std(axis=1) / np.sqrt(Vg.shape[1]) / Vg_static
        ax.errorbar(T_list, ratio, yerr=sem, marker='o', capsize=3,
                    color=colors[c], label=labels[c])

    # simulated σ²=0 baseline, also divided by the per-T analytic baseline
    if os.path.exists(BASE_FILE):
        Vg0    = np.load(BASE_FILE)[a1_name]    # (len(T_list), rep)
        ratio0 = Vg0.mean(axis=1) / Vg_static
        sem0   = Vg0.std(axis=1) / np.sqrt(Vg0.shape[1]) / Vg_static
        ax.errorbar(T_list, ratio0, yerr=sem0, marker='s', capsize=3,
                    color='k', ls='-', lw=1.6,
                    label=r'simulated $V_g|_{\sigma^2=0}$ / analytic')
    else:
        print(f"[note] {BASE_FILE} not found; σ²=0 baseline curve omitted.")

    # analytic multi-trait Latter–Bulmer baseline is 1 by construction (all T)
    ax.axhline(1.0, color='k', ls='--', lw=1,
               label=r'analytic $4L\mu V_s/T$  (=1)')

    ax.set_xscale('log')
    ax.set_xticks(T_list)
    ax.set_xticklabels([str(int(T)) for T in T_list])
    ax.set_xlabel('number of trait dimensions  T')
    ax.set_ylabel(r'$V_g(T)\,/\,(4L\mu V_s/T)$')
    ax.set_title(
        rf'Genetic variance relative to the analytic static baseline'
        '\n'
        rf'(analytic $4L\mu V_s/T$, $4L\mu V_s={Vg_static_1:.4g}$;  '
        rf'A~{dist_name}, dir={dir_name}, $a_t$~{a1_name}, $a^2$={a2:.2f})'
    )
    ax.legend(fontsize=8, loc='best')
    fig.tight_layout()

    out = f'Vg_ratio_over_T_{tag}.pdf'
    fig.savefig(out)
    print(f"Saved {out}")

plt.show()
