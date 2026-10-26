# -*- coding: utf-8 -*-
r"""
The G matrix and the quadratic form  Q = sum_{t,r} G_tr delta_t delta_r = delta^T G delta,
over the number of traits T.

PLOTTING ONLY -- this script runs no simulation.  It reads the cross_term_data_*.npz
files written by sweep_T_4cases_violin.py (tracked replicate, post burn-in).

How each quantity is obtained from the saved data:
  * Q on the REC_EVERY grid, exactly:  Q = sum_m delta_m (own_m + cross_m),
    since own + cross = (G delta)_m.
  * G on the TRAIT_REC_EVERY grid (G itself is not saved):  the file stores
    ad = a_{im} delta_m and tp = p_i, and delta at the same generation, so
        a_{im} = ad_{im} / delta_m,   G_tr = 2 sum_i p_i q_i a_{it} a_{ir}.
    ad is float32, so G is good to ~1e-6 relative; diag(G) is checked against the
    saved V_g,t and the worst relative mismatch is printed.

Output (in the results folder):
  G_matrix_<dir>_case<c>_a2_<a2>.pdf   rows: one snapshot / time mean / time variance
                                       of G_tr; one column per T  (matshow heat maps)
  quadform_GDD_<dir>_a2_<a2>.pdf       E[Q], Var(Q), Var(Q)/E[Q]^2 vs T, all cases
  quadform_GDD_trace_<dir>_a2_<a2>.pdf Q(generation), rows = cases, cols = T
"""

import os
import glob
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── output directory (must match sweep_T_4cases_violin.py / var_delta_sq_figs.py) ──
RESULTS_DIR = 'results Sep 15'
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', RESULTS_DIR)
def out(name):  return os.path.join(OUTDIR, name)

CASE_LIST = ['A', 'B', 'C', 'D', 'E', 'F']
case_labels = {
    'A': r'A: $\Sigma_{ii}=\sigma^2,\ \Sigma_{ij}>0$',
    'B': r'B: $\Sigma_{ii}=\sigma^2,\ \Sigma_{ij}<0$',
    'C': r'C: $\Sigma_{ii}=\sigma^2/T,\ \Sigma_{ij}>0$',
    'D': r'D: $\Sigma_{ii}=\sigma^2/T,\ \Sigma_{ij}<0$',
    'E': r'E: $\Sigma=(\sigma^2/T)\,I$',
    'F': r'F: $\Sigma=\sigma^2 I$',
}
case_colors = {'A': 'C0', 'B': 'C3', 'C': 'C2', 'D': 'C1', 'E': 'C4', 'F': 'C5'}
N_BLOCKS = 10                               # blocks for the jackknife error bars


def load(path):
    z = np.load(path)
    d = {k: z[k] for k in z.files}
    for k in ('T', 'L', 'N', 'BURN_IN', 'REC_EVERY', 'TRACK_REP', 'rep'):
        if k in d:
            d[k] = int(d[k])
    for k in ('V_s', 'a2', 'sigma_e2'):
        if k in d:
            d[k] = float(d[k])
    for k in ('dir_name', 'case', 'dist_name'):
        if k in d:
            d[k] = str(d[k])
    return d


def G_series(d):
    """G (n_tr, T, T) on the TRAIT_REC_EVERY grid, rebuilt from ad / delta and tp."""
    gen   = np.asarray(d['gen'])
    idx   = np.searchsorted(gen, d['tgen'])
    assert np.array_equal(gen[idx], d['tgen']), 'tgen is not on the REC_EVERY grid'
    dl    = np.asarray(d['delta'], dtype=float)[idx]          # (n_tr, T)
    ad    = np.asarray(d['ad'],    dtype=float)                # (n_tr, L, T)
    safe  = np.where(dl == 0, 1.0, dl)
    a     = np.where(dl[:, None, :] == 0, 0.0, ad / safe[:, None, :])
    tp    = np.asarray(d['tp'], dtype=float)
    pq    = tp * (1 - tp)                                      # (n_tr, L)
    G     = 2 * np.einsum('kl,klt,klr->ktr', pq, a, a)
    Vg    = np.asarray(d['Vg'], dtype=float)[idx]
    diag  = np.einsum('ktt->kt', G)
    relerr = np.abs(diag - Vg).max() / max(np.abs(Vg).max(), 1e-300)
    return G, relerr


def block_se(x, fn):
    """Block-jackknife standard error of fn(x) over N_BLOCKS consecutive blocks."""
    blocks = np.array_split(np.arange(len(x)), N_BLOCKS)
    reps = np.array([fn(np.delete(x, b)) for b in blocks])
    B = len(reps)
    return float(np.sqrt((B - 1) / B * np.sum((reps - reps.mean()) ** 2)))


def heat(ax, fig, M, title, symmetric):
    """One matshow heat map with its own colour bar."""
    if symmetric:
        lim = np.abs(M).max() or 1.0
        im = ax.matshow(M, cmap='RdBu_r', vmin=-lim, vmax=lim)
    else:
        im = ax.matshow(M, cmap='viridis')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=9)
    T = M.shape[0]
    ticks = np.unique(np.linspace(0, T - 1, min(T, 5)).astype(int))
    ax.set_xticks(ticks); ax.set_xticklabels(ticks + 1, fontsize=7)
    ax.set_yticks(ticks); ax.set_yticklabels(ticks + 1, fontsize=7)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlabel('trait $r$', fontsize=8); ax.set_ylabel('trait $t$', fontsize=8)


def figure_G(Gs, T_list, case, dir_name, a2, hdr):
    """Rows: last snapshot, time mean, time variance of G_tr.  Columns: T."""
    nT = len(T_list)
    fig, axes = plt.subplots(3, nT, figsize=(4.0 * nT, 11.5), squeeze=False)
    for j, T in enumerate(T_list):
        G, tgen = Gs[T]
        heat(axes[0, j], fig, G[-1], f'T = {T}: snapshot, gen {tgen[-1]}', True)
        heat(axes[1, j], fig, G.mean(axis=0), f'T = {T}: time mean', True)
        heat(axes[2, j], fig, G.var(axis=0), f'T = {T}: time variance', False)
    for i, lab in enumerate([r'$G_{tr}$ (one generation)',
                             r'$\langle G_{tr}\rangle_\tau$',
                             r'Var$_\tau(G_{tr})$']):
        axes[i, 0].annotate(lab, xy=(-0.45, 0.5), xycoords='axes fraction',
                            rotation=90, va='center', ha='center', fontsize=11)
    fig.suptitle(r'G MATRIX $G_{tr}=2\sum_i p_iq_i\,a_{it}a_{ir}$ vs number of traits, '
                 f'{case_labels[case]}\n'
                 r'red/blue = positive/negative (symmetric scale per panel); '
                 'tracked replicate, post burn-in, every 100 gens\n'
                 f'{hdr}', fontsize=10, y=1.00)
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.04, top=0.88,
                        wspace=0.45, hspace=0.35)
    fname = f'G_matrix_{dir_name}_case{case}_a2_{a2:.2f}.pdf'
    fig.savefig(out(fname), bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}")


def figure_Q(res, T_list, cases, dir_name, a2, hdr):
    """E[Q], Var(Q) and Var(Q)/E[Q]^2 vs T."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))

    def setup(ax, ylabel, title, logy=True):
        ax.set_xscale('log')
        if logy:
            ax.set_yscale('log')
        ax.set_xticks(T_list); ax.set_xticklabels([str(t) for t in T_list])
        ax.set_xlabel('Number of traits $T$'); ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3); ax.set_title(title, fontsize=10)

    setup(axes[0], r'$E[Q]$',
          r'(a) mean of $Q=\sum_{t,r}G_{tr}\delta_t\delta_r$')
    setup(axes[1], r'Var$(Q)$', r'(b) variance of $Q$ over generations')
    setup(axes[2], r'Var$(Q)\,/\,E[Q]^2$', r'(c) relative fluctuation of $Q$')
    for c in cases:
        col = case_colors[c]
        for ax, key in zip(axes, ('E', 'Var', 'rel')):
            ax.errorbar(T_list, [res[(c, T)][key] for T in T_list],
                        yerr=[res[(c, T)]['se_' + key] for T in T_list],
                        marker='o', color=col, capsize=3, lw=1.5, label=case_labels[c])
    axes[0].legend(fontsize=7)
    fig.suptitle(r'QUADRATIC FORM $Q=\vec\delta^{\,T}G\,\vec\delta$ vs NUMBER OF TRAITS '
                 '(tracked replicate, post burn-in)'
                 f'\n{hdr}', fontsize=10, y=1.02)
    fig.subplots_adjust(left=0.05, right=0.99, bottom=0.11, top=0.80, wspace=0.28)
    fname = f'quadform_GDD_{dir_name}_a2_{a2:.2f}.pdf'
    fig.savefig(out(fname), bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}")


def figure_Q_trace(traces, T_list, cases, dir_name, a2, hdr):
    """Q(generation); rows = cases, columns = T."""
    nT, nC = len(T_list), len(cases)
    fig, axes = plt.subplots(nC, nT, figsize=(3.6 * nT, 2.4 * nC), squeeze=False)
    for i, c in enumerate(cases):
        for j, T in enumerate(T_list):
            ax = axes[i, j]
            g, Q = traces[(c, T)]
            ax.plot(g, Q, color=case_colors[c], lw=0.4)
            ax.axhline(Q.mean(), color='k', lw=1.0, label='time mean')
            ax.set_yscale('log')
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.set_title(f'T = {T}', fontsize=10)
            if j == 0:
                ax.set_ylabel(f'case {c}\n$Q$', fontsize=9)
            if i == nC - 1:
                ax.set_xlabel('generation', fontsize=8)
            if i == 0 and j == 0:
                ax.legend(fontsize=6)
    fig.suptitle(r'$Q=\sum_{t,r}G_{tr}\delta_t\delta_r$ over generations '
                 f'(post burn-in)\n{hdr}', fontsize=10, y=1.00)
    fig.tight_layout()
    fname = f'quadform_GDD_trace_{dir_name}_a2_{a2:.2f}.pdf'
    fig.savefig(out(fname), bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}")


# ── main ──────────────────────────────────────────────────────────────────────
PATTERN = re.compile(r'cross_term_data_(?P<dir>\w+?)_T(?P<T>\d+)_'
                     r'case(?P<case>[A-F])_a2_(?P<a2>[\d.]+)\.npz$')

files = sorted(glob.glob(out('cross_term_data_*.npz')))
if not files:
    raise SystemExit(f'No cross_term_data_*.npz in {OUTDIR}.\n'
                     'Run sweep_T_4cases_violin.py first -- it is the only script '
                     'that simulates.')

groups = {}
for f in files:
    mt = PATTERN.search(os.path.basename(f))
    if mt and mt['case'] in CASE_LIST:
        groups.setdefault((mt['dir'], float(mt['a2'])), []).append((int(mt['T']),
                                                                    mt['case'], f))

for (dir_name, a2), entries in sorted(groups.items()):
    print(f"\n############## dir = {dir_name}, a2 = {a2:.2f} ##############")
    print(f"{'T':>4} {'case':>4} {'E[Q]':>10} {'Var(Q)':>11} "
          f"{'Var/E^2':>8} {'diag(G) err':>11}")
    T_seen = sorted({T for T, _, _ in entries})
    res, traces, Gs, hdr = {}, {}, {}, ''

    for T, case, f in sorted(entries):
        d = load(f)
        hdr = (rf"L={d['L']}, N={d['N']}, $\mu$={float(d['mu']):g}, "
               rf"$V_s$={d['V_s']:g}, $a^2$={d['a2']:g}, dir={d['dir_name']}, "
               rf"$\sigma^2$={d['sigma_e2']:g}, $\theta$={float(d['theta']):g}, "
               f"generations={int(d['maxiter'])}, burn-in={d['BURN_IN']}, "
               f"error bars: {N_BLOCKS}-block jackknife")

        # Q on the REC_EVERY grid
        m   = d['gen'] >= d['BURN_IN']
        dl  = np.asarray(d['delta'], dtype=float)[m]
        Gd  = np.asarray(d['own'], dtype=float)[m] + np.asarray(d['cross'], dtype=float)[m]
        Q   = (dl * Gd).sum(axis=1)
        rel = lambda x: x.var() / x.mean() ** 2
        res[(case, T)] = dict(E=Q.mean(), Var=Q.var(), rel=rel(Q),
                              se_E=block_se(Q, np.mean), se_Var=block_se(Q, np.var),
                              se_rel=block_se(Q, rel))
        traces[(case, T)] = (d['gen'][m], Q)

        # G on the TRAIT_REC_EVERY grid (already post burn-in)
        G, relerr = G_series(d)
        Gs.setdefault(case, {})[T] = (G, d['tgen'])

        r = res[(case, T)]
        print(f"{T:>4} {case:>4} {r['E']:>10.4g} {r['Var']:>11.4g} "
              f"{r['rel']:>8.3g} {relerr:>11.2g}")

    cases = [c for c in CASE_LIST if all((c, T) in res for T in T_seen)]
    for c in CASE_LIST:
        if c not in cases:
            print(f"  [note] case {c} not available at every T; omitted")
    for c in cases:
        figure_G(Gs[c], T_seen, c, dir_name, a2, hdr)
    if cases:
        figure_Q(res, T_seen, cases, dir_name, a2, hdr)
        figure_Q_trace(traces, T_seen, cases, dir_name, a2, hdr)

print("\nDone.")
