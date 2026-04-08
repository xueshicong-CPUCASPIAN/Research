# -*- coding: utf-8 -*-
"""
n-dimensional figures analogous to Figure 1 and Figure 2 in the paper.

Figure 1 (time series):
  Panel A: allele frequency trajectories (sigma_e2=0 vs sigma_e2=1e-2)
  Panel B: heritability h² = Vg/(1+Vg) over time
  Panel C: ||delta_t||_2 = ||opt - zbar||_2 over time
  Requires: hist_000_nd.txt, hist_001_nd.txt,
            Vg_hist_000_nd.txt, Vg_hist_001_nd.txt,
            delta_norm_001_nd.txt
  (run simulate_trajectory_ndim.py first)

Figure 2 (violin plots):
  Heritability h² vs fluctuation intensity sigma² for different n_traits
  Requires: Vg_sims_n_dimension
  (run simulate_mpi n_dimension.py first)
"""

import numpy as np
import matplotlib.pyplot as plt

# ── helper: split trajectories at fixation (same as figures.py) ──────────────
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
# FIGURE 1 — time series
##############################################################################
print("Plotting Figure 1 (n-D time series)...")

hist0       = np.loadtxt('hist_000_nd.txt')        # (maxiter, L)
Vg0         = np.loadtxt('Vg_hist_000_nd.txt')     # (maxiter,)
hist1       = split_traj(np.loadtxt('hist_001_nd.txt'))
Vg1         = np.loadtxt('Vg_hist_001_nd.txt')     # (maxiter,)
delta_norm1 = np.loadtxt('delta_norm_001_nd.txt')  # (maxiter,)

t1 = 95000
t2 = 100000

fig, axs = plt.subplots(3, 1, figsize=[3, 6])

# Panel A: allele frequency trajectories
axs[0].plot(hist1[t1:t2, :], 'gray', linewidth=0.5)
axs[0].plot(hist0[t1:t2, :], 'k',    linewidth=0.5)
axs[0].set_ylim([0, 1.])
axs[0].set_xlim([0, t2 - t1])
axs[0].set_xticklabels([])
axs[0].set_ylabel(r'Frequency', fontsize=10)
axs[0].annotate(r'$A$', [0.92, 0.87], xycoords='axes fraction', fontsize=14)

# Panel B: heritability h² = Vg/(1+Vg)
axs[1].plot(Vg0[t1:t2] / (1 + Vg0[t1:t2]), 'k',    label=r'$\sigma^2=0$')
axs[1].plot(Vg1[t1:t2] / (1 + Vg1[t1:t2]), 'gray', label=r'$\sigma^2=10^{-2}$')
axs[1].set_ylim([0, 1])
axs[1].set_xlim([0, t2 - t1])
axs[1].set_xticklabels([])
axs[1].set_ylabel(r'Heritability', fontsize=10)
axs[1].annotate(r'$B$', [0.92, 0.87], xycoords='axes fraction', fontsize=14)
axs[1].legend(loc='upper left', fontsize=5)

# inset histogram of h²
inset = axs[1].inset_axes([0.42, 0.69, 0.3, 0.30])
inset.hist(Vg1 / (1 + Vg1), bins=20, color='gray')
inset.hist(Vg0 / (1 + Vg0), color='k')
inset.set_yticklabels([])
inset.set_yticks([])
inset.set_xticks([0, 0.6])
inset.set_xticklabels([0, 0.6], fontsize=8)
inset.set_xlabel(r'$h^2$', labelpad=-10)

# Panel C: ||delta_t||_2
axs[2].plot(delta_norm1[t1:t2], 'gray')
axs[2].axhline(y=0, color='k')
axs[2].set_xlim([0, t2 - t1])
axs[2].set_ylabel(r'$\|\delta_t\|$', fontsize=10)
axs[2].set_xlabel(r'Generations')
axs[2].annotate(r'$C$', [0.92, 0.87], xycoords='axes fraction', fontsize=14)

plt.tight_layout()
plt.savefig('timeseries_ndim.pdf', bbox_inches='tight')
print("Saved timeseries_ndim.pdf")


##############################################################################
# FIGURE 2 — heritability vs sigma² for different n_traits
##############################################################################
print("Plotting Figure 2 (n-D violin plots)...")

fname = "Vg_sims_n_dimension"
with open(fname, 'r') as fin:
    params = eval(fin.readline()[2:-1])
    # parameter format: L, sigma_e2, N, V_s, mu, a2, theta, n_traits, rep

Vg_sims = np.loadtxt(fname)

offset = 1e-5
N_plot = 10000
a2_plot = 0.1
V_s_plot = 5  # fix V_s to one value for clarity; change to 20 if preferred

# collect all n_traits values present in the output
all_n_traits = sorted(set(p[7] for p in params))
colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(all_n_traits)))

fig, ax = plt.subplots(figsize=[5, 4])

for color, nt in zip(colors, all_n_traits):
    indices = [i for i, p in enumerate(params)
               if p[2] == N_plot and p[-3] == a2_plot and p[3] == V_s_plot and p[7] == nt]

    for i in indices:
        L, sigma_e2, N, V_s, mu, a2, theta, n_traits, rep = params[i]
        h2 = Vg_sims[i] / (Vg_sims[i] + 1)
        parts = ax.violinplot(
            h2,
            positions=[np.log10(sigma_e2 + offset)],
            widths=0.25, showmeans=True)
        for pc in parts['bodies']:
            pc.set_color(color)
            pc.set_alpha(0.5)
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
            parts[partname].set_color(color)

# legend
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
