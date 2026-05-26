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

import numpy as np
import matplotlib.pyplot as plt

a2 = 0.01   # must match sweep_T_4cases_hist.py

d = np.load('hist_T_4cases_data_a_0.01.npz')
T_list = d['T_list']
cases  = ['A', 'B', 'C', 'D']

colors = {'A': 'C0', 'B': 'C3', 'C': 'C2', 'D': 'C1'}
labels = {
    'A': r'A: $\Sigma_{ii}=\sigma^2,\ \Sigma_{ij}=+\sigma^2$',
    'B': r'B: $\Sigma_{ii}=\sigma^2,\ \Sigma_{ij}=-\sigma^2$',
    'C': r'C: $\Sigma_{ii}=\sigma^2/T,\ \Sigma_{ij}=+\sigma^2/T$',
    'D': r'D: $\Sigma_{ii}=\sigma^2/T,\ \Sigma_{ij}=-\sigma^2/T$',
}

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
ax.plot(T_list, a2 / np.array(T_list), 'k--', lw=1, label=r'$a^2/T$ (theory)')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'$T$')
ax.set_ylabel(r'$E[a_{1,l}^2]$')
ax.set_title(r'(1)  $E[a_{1,l}^2]$  vs $T$' '\n(slope $-1$, no case dependence)')
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
ax.set_title(r'(2)  $E[p_l]$  vs $T$')
ax.grid(True, which='both', alpha=0.3)

# (3) E[p_l (1 - p_l)]  (drives Vg)
ax = axes[2]
for c in cases:
    ax.errorbar(T_list, mean_pq[c], yerr=sem_pq[c],
                marker='o', color=colors[c], label=labels[c], capsize=3)
ax.set_xscale('log')
ax.set_xlabel(r'$T$')
ax.set_ylabel(r'$E[p_l (1-p_l)]$')
ax.set_title(r'(3)  $E[p_l (1-p_l)]$  vs $T$' '\n(direct driver of $V_g$)')
ax.grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.savefig('hist_T_4cases_summary_a_0.01.pdf', bbox_inches='tight')
print("Saved hist_T_4cases_summary_a_0.01.pdf")

# Also print the numbers
print("\n=== E[a_{1,l}^2] ===")
print(f"{'T':>6} | " + "  ".join(f"{c:>10}" for c in cases) + "  |  a^2/T")
for ti, T in enumerate(T_list):
    row = "  ".join(f"{mean_a1sq[c][ti]:10.3e}" for c in cases)
    print(f"{T:>6} | {row}  |  {a2/T:.3e}")

print("\n=== E[p_l] ===")
print(f"{'T':>6} | " + "  ".join(f"{c:>10}" for c in cases))
for ti, T in enumerate(T_list):
    row = "  ".join(f"{mean_p[c][ti]:10.4f}" for c in cases)
    print(f"{T:>6} | {row}")

print("\n=== E[p_l (1 - p_l)] ===")
print(f"{'T':>6} | " + "  ".join(f"{c:>10}" for c in cases))
for ti, T in enumerate(T_list):
    row = "  ".join(f"{mean_pq[c][ti]:10.4e}" for c in cases)
    print(f"{T:>6} | {row}")
