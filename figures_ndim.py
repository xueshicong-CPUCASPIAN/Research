# -*- coding: utf-8 -*-
"""
n-dimensional figures with FULL covariance matrix Sigma for the optimum noise.

Figure 1 (4-case time series): produces 4 columns (cases A, B, C, D), each with
3 rows (allele freq, heritability, ||delta||).
  Requires: hist_A_nd.txt, ..., hist_D_nd.txt, Vg_hist_*, delta_norm_*, plus
            hist_000_nd.txt, Vg_hist_000_nd.txt for the constant-optimum baseline
  Run simulate_trajectory_ndim.py first.

Figure 2 (violin plots): heritability h^2 vs sigma^2 for different n_traits
  Requires: Vg_sims_n_dimension
  (run simulate_mpi n_dimension.py first)
"""

import numpy as np
import matplotlib.pyplot as plt

# ── helper: split trajectories at fixation ───────────────────────────────────
def split_traj(hist):
    temp = []
    hist = np.array(hist.transpose())
    for loc, _ in enumerate(hist):
        _ = np.array(_)
        ones_pos = np.arange(len(_))[_ == 1]
        ones_pos = np.concatenate(([0], ones_pos, [len(_)]))
        if len(ones_pos) > 2:
            for i in range(len(ones_pos) - 1):
                traj = np.concatenate((
                    np.zeros(ones_pos[i] + 1),
                    _[ones_pos[i] + 1:ones_pos[i + 1]],
                    np.ones(len(_) - ones_pos[i + 1])
                ))
                if i == 0:
                    hist[loc] = traj
                else:
                    temp.append(traj)
    if len(temp) == 0:
        return hist.transpose()
    return np.concatenate((hist, np.array(temp))).transpose()


##############################################################################
# FIGURE 1 — time series for 4 covariance-matrix cases
##############################################################################
print("Plotting Figure 1 (4-case time series)...")

# Case descriptions for column titles
case_titles = {
    'A': r'$\Sigma_{ii}=\sigma^2,\ \Sigma_{ij}=+\sigma^2$',
    'B': r'$\Sigma_{ii}=\sigma^2,\ \Sigma_{ij}=-\sigma^2$',
    'C': r'$\Sigma_{ii}=\sigma^2/T,\ \Sigma_{ij}=+\sigma^2/T$',
    'D': r'$\Sigma_{ii}=\sigma^2/T,\ \Sigma_{ij}=-\sigma^2/T$',
}

# load constant-optimum baseline (same for all cases)
hist0 = np.loadtxt('hist_000_nd.txt')
Vg0   = np.loadtxt('Vg_hist_000_nd.txt')

t1 = 0
t2 = 5000

fig, axs = plt.subplots(3, 4, figsize=[14, 7], sharey='row')

for col, label in enumerate(['A', 'B', 'C', 'D']):
    hist = split_traj(np.loadtxt(f'hist_{label}_nd.txt'))
    Vg   = np.loadtxt(f'Vg_hist_{label}_nd.txt')
    dn   = np.loadtxt(f'delta_norm_{label}_nd.txt')

    # Panel A row: allele frequencies
    axs[0, col].plot(hist[t1:t2, :], 'gray', linewidth=0.5)
    axs[0, col].plot(hist0[t1:t2, :], 'k', linewidth=0.5)
    axs[0, col].set_ylim([0, 1])
    axs[0, col].set_xlim([0, t2 - t1])
    axs[0, col].set_xticklabels([])
    axs[0, col].set_title(f"Case {label}\n{case_titles[label]}", fontsize=10)
    if col == 0:
        axs[0, col].set_ylabel('Frequency', fontsize=11)

    # Panel B row: heritability
    axs[1, col].plot(Vg0[t1:t2] / (1 + Vg0[t1:t2]), 'k',    label=r'$\sigma^2=0$')
    axs[1, col].plot(Vg[t1:t2]  / (1 + Vg[t1:t2]),  'gray', label=r'$\sigma^2=10^{-2}$')
    axs[1, col].set_ylim([0, 1])
    axs[1, col].set_xlim([0, t2 - t1])
    axs[1, col].set_xticklabels([])
    if col == 0:
        axs[1, col].set_ylabel(r'Heritability $h^2$', fontsize=11)
        axs[1, col].legend(loc='upper left', fontsize=7)

    # Panel C row: ||delta||
    axs[2, col].plot(dn[t1:t2], 'gray')
    axs[2, col].axhline(y=0, color='k', linewidth=0.5)
    axs[2, col].set_xlim([0, t2 - t1])
    axs[2, col].set_xlabel('Generations', fontsize=11)
    if col == 0:
        axs[2, col].set_ylabel(r'$\|\delta_t\|$', fontsize=11)

plt.tight_layout()
plt.savefig('timeseries_ndim.pdf', bbox_inches='tight')
print("Saved timeseries_ndim.pdf")


##############################################################################
# FIGURE 2 — heritability vs sigma^2 for different n_traits  (unchanged)
##############################################################################
print("Plotting Figure 2 (n-D violin plots)...")

fname = "Vg_sims_n_dimension"
with open(fname, 'r') as fin:
    params = eval(fin.readline()[2:-1])

Vg_sims = np.loadtxt(fname)

offset   = 1e-5
N_plot   = 10000
a2_plot  = 0.1
V_s_plot = 5
L = params[0][0]

all_n_traits = sorted(set(p[7] for p in params))
colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(all_n_traits)))

fig, ax = plt.subplots(figsize=[5, 4])
n_nt = len(all_n_traits)
displacements = np.linspace(-0.15, 0.15, n_nt)

for color, nt, displace in zip(colors, all_n_traits, displacements):
    indices = [i for i, p in enumerate(params)
               if p[2] == N_plot and p[5] == a2_plot and p[3] == V_s_plot and p[7] == nt]
    for i in indices:
        L, sigma_e2, N, V_s, mu, a2, theta, n_traits, rep = params[i]
        h2 = Vg_sims[i] / (Vg_sims[i] + 1)
        parts = ax.violinplot(h2,
                              positions=[np.log10(sigma_e2 + offset) + displace],
                              widths=0.08, showmeans=True)
        for pc in parts['bodies']:
            pc.set_color(color)
            pc.set_alpha(0.5)
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
            parts[partname].set_color(color)

handles = [plt.matplotlib.patches.Patch(color=colors[i], alpha=0.7,
           label=f'$T={int(nt)}$') for i, nt in enumerate(all_n_traits)]
ax.legend(handles=handles, fontsize=9, title='Trait dimensions')
ax.set_ylim([0, 1])
ax.set_ylabel(r'Heritability $h^2$', fontsize=12)
ax.set_xlabel(r'Fluctuation intensity $\sigma^2$', fontsize=12)
ax.set_title(f'$L={int(L)},\\ V_s={V_s_plot},\\ N={N_plot}$', fontsize=11)
tick_positions = [np.log10(s + offset) for s in [0, 1e-4, 1e-3, 1e-2]]
tick_labels = [r'$0$', r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$']
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, fontsize=10)

plt.tight_layout()
plt.savefig('violinplot_ndim.pdf', bbox_inches='tight')
print("Saved violinplot_ndim.pdf")
