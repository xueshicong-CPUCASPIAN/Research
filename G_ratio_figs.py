# -*- coding: utf-8 -*-
r"""
Diagonal against off-diagonal of the G matrix, per trait and over the number of traits T:

    f(i) = < G_ii / sum_{j != i} G_ij >        (average over time),   i = trait

and its trait average  (1/T) sum_i f(i),  for L = 100 and L = 1000.

PLOTTING ONLY -- this script runs no simulation.  It reads the cross_term_data_*.npz
files written by sweep_T_4cases_violin.py (tracked replicate, post burn-in) from each
folder in RESULTS_DIRS and writes the figures back into the SAME folder.

Why.  The response of trait i is (G delta)_i = G_ii delta_i + sum_{j != i} G_ij delta_j:
its own genetic variance plus the summed genetic covariances with the other traits.
The ratio says how much the diagonal dominates row i of G.  T = 1 has no off-diagonal
and is skipped.

Four versions of f(i) are drawn, because the plain one is fragile:
  signed, mean of ratio    < G_ii / sum_{j!=i} G_ij >            (as written in the notes)
  signed, ratio of means   < G_ii > / < sum_{j!=i} G_ij >
  abs,    mean of ratio    < G_ii / sum_{j!=i} |G_ij| >
  abs,    ratio of means   < G_ii > / < sum_{j!=i} |G_ij| >
The signed off-diagonal sum can pass through zero, so the signed mean of ratio can be
dominated by a few generations with a tiny denominator; the other three are the checks.
Generations where the denominator is exactly 0 (e.g. no segregating locus) are dropped
from the mean of ratio, and the fraction dropped is printed.

G is rebuilt on the TRAIT_REC_EVERY grid exactly as in G_quadform_figs.py:
    a_{im} = ad_{im} / delta_m,   G_tr = 2 sum_i p_i q_i a_{it} a_{ir}.

Output (per folder, per (direction, a2)):
  G_ratio_by_trait_<dir>_a2_<a2>.pdf   f(i) against trait index i; rows = the four
                                       versions, cols = T, one line per case
  G_ratio_over_T_<dir>_a2_<a2>.pdf     (1/T) sum_i f(i) against T; one panel per version,
                                       one line per case, block-jackknife error bars
"""

import os
import glob
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── data / output folders, one per L (figures go into the folder the data came from) ──
RESULTS_DIRS = {100: 'results sep 15 OLD', 1000: 'results_Nov_2_L_1000'}
HERE = os.path.dirname(os.path.abspath(__file__))

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

# key -> (use |G_ij| ?, ratio of time means ?, row label)
VERSIONS = {
    'signed_mr': (False, False, r'$\langle G_{ii}/\sum_{j\neq i}G_{ij}\rangle$'),
    'signed_rm': (False, True,  r'$\langle G_{ii}\rangle/\langle\sum_{j\neq i}G_{ij}\rangle$'),
    'abs_mr':    (True,  False, r'$\langle G_{ii}/\sum_{j\neq i}|G_{ij}|\rangle$'),
    'abs_rm':    (True,  True,  r'$\langle G_{ii}\rangle/\langle\sum_{j\neq i}|G_{ij}|\rangle$'),
}


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
    return 2 * np.einsum('kl,klt,klr->ktr', pq, a, a)


def f_of_i(diag, off, ratio_of_means):
    """f(i), shape (T,), from the time series diag, off of shape (n, T)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        if ratio_of_means:
            den = off.mean(axis=0)
            return np.where(den != 0, diag.mean(axis=0) / den, np.nan)
        r = np.where(off != 0, diag / off, np.nan)             # (n, T)
        ok = np.isfinite(r)
        cnt = ok.sum(axis=0)
        return np.where(cnt > 0, np.where(ok, r, 0.0).sum(axis=0) / np.maximum(cnt, 1),
                        np.nan)


def trait_avg(diag, off, ratio_of_means):
    f = f_of_i(diag, off, ratio_of_means)
    return np.nanmean(f) if np.isfinite(f).any() else np.nan


def block_se(diag, off, ratio_of_means):
    """Block-jackknife standard error of the trait average over N_BLOCKS time blocks."""
    n = len(diag)
    blocks = np.array_split(np.arange(n), min(N_BLOCKS, n))
    reps = np.array([trait_avg(np.delete(diag, b, axis=0), np.delete(off, b, axis=0),
                               ratio_of_means) for b in blocks])
    reps = reps[np.isfinite(reps)]
    B = len(reps)
    if B < 2:
        return np.nan
    return float(np.sqrt((B - 1) / B * np.sum((reps - reps.mean()) ** 2)))


def set_yscale(ax, key, abs_ylim=None):
    """Signed versions can be negative: symlog.  Abs versions are positive: log, and
    every abs panel (both abs versions, every T, every L) gets the same limits."""
    if key.startswith('signed'):
        ax.set_yscale('symlog', linthresh=1.0)
        ax.axhline(0, color='0.5', lw=0.6)
    else:
        ax.set_yscale('log')
        if abs_ylim is not None:
            ax.set_ylim(*abs_ylim)
    ax.axhline(1, color='0.5', lw=0.6, ls=':')                 # diagonal = off-diagonal sum


def figure_by_trait(fi, T_plot, cases, dir_name, a2, hdr, out, abs_ylim):
    """f(i) against trait index i.  Rows = versions, cols = T."""
    nr, nc = len(VERSIONS), len(T_plot)
    fig, axes = plt.subplots(nr, nc, figsize=(4.0 * nc, 3.0 * nr), squeeze=False)
    for r, (key, (_, _, lab)) in enumerate(VERSIONS.items()):
        for c, T in enumerate(T_plot):
            ax = axes[r, c]
            x = np.arange(1, T + 1)
            for case in cases:
                if (case, T) in fi:
                    ax.plot(x, fi[(case, T)][key], color=case_colors[case], lw=1.0,
                            marker='o' if T <= 20 else None, ms=3)
            set_yscale(ax, key, abs_ylim)
            ax.grid(alpha=0.3)
            if T <= 20:
                ax.set_xticks(x if T <= 5 else np.arange(0, T + 1, 5)[1:])
            if r == 0:
                ax.set_title(f'T = {T}', fontsize=10)
            if c == 0:
                ax.set_ylabel(lab, fontsize=10)
            if r == nr - 1:
                ax.set_xlabel('trait $i$')
    handles = [plt.Line2D([], [], color=case_colors[c], lw=2, label=case_labels[c])
               for c in cases]
    H = fig.get_figheight()
    fig.suptitle(r'$f(i)$: DIAGONAL vs OFF-DIAGONAL SUM OF ROW $i$ OF $G$, by trait'
                 ' (tracked replicate, post burn-in; dotted line = 1)'
                 f'\n{hdr}', fontsize=10, y=0.995)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1 - 0.65 / H),
               ncol=len(handles), fontsize=8, frameon=False)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.05, top=1 - 1.25 / H,
                        hspace=0.25, wspace=0.25)
    fname = f'G_ratio_by_trait_{dir_name}_a2_{a2:.2f}.pdf'
    fig.savefig(out(fname), bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}")


def figure_over_T(avg, T_plot, cases, dir_name, a2, hdr, out, abs_ylim):
    """(1/T) sum_i f(i) against T.  One panel per version."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), squeeze=False)
    for ax, (key, (_, _, lab)) in zip(axes.ravel(), VERSIONS.items()):
        for case in cases:
            Ts = [T for T in T_plot if (case, T) in avg]
            ax.errorbar(Ts, [avg[(case, T)][key][0] for T in Ts],
                        yerr=[avg[(case, T)][key][1] for T in Ts],
                        marker='o', color=case_colors[case], capsize=3, lw=1.5,
                        label=case_labels[case])
        ax.set_xscale('log')
        set_yscale(ax, key, abs_ylim)
        ax.set_xticks(T_plot); ax.set_xticklabels([str(t) for t in T_plot])
        ax.set_xlabel('Number of traits $T$')
        ax.set_ylabel(r'$\frac{1}{T}\sum_i f(i)$')
        ax.set_title(r'$f(i)=$' + lab, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(r'TRAIT-AVERAGED DIAGONAL / OFF-DIAGONAL RATIO OF $G$ vs NUMBER OF TRAITS'
                 f' (tracked replicate, post burn-in; dotted line = 1)\n{hdr}',
                 fontsize=10, y=0.995)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.06, top=0.90,
                        hspace=0.28, wspace=0.22)
    fname = f'G_ratio_over_T_{dir_name}_a2_{a2:.2f}.pdf'
    fig.savefig(out(fname), bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {fname}")


# ── main ──────────────────────────────────────────────────────────────────────
PATTERN = re.compile(r'cross_term_data_(?P<dir>\w+?)_T(?P<T>\d+)_'
                     r'case(?P<case>[A-F])_a2_(?P<a2>[\d.]+)\.npz$')

jobs = []           # figures are drawn after all folders are read (shared y-range)
for L_expected, results_dir in RESULTS_DIRS.items():
    outdir = os.path.join(HERE, '..', results_dir)
    out = lambda name, _d=outdir: os.path.join(_d, name)
    print(f"\n==================== L = {L_expected}:  {results_dir} ====================")

    files = sorted(glob.glob(out('cross_term_data_*.npz')))
    if not files:
        print(f"  [skip] no cross_term_data_*.npz in {outdir}")
        continue

    groups = {}
    for f in files:
        mt = PATTERN.search(os.path.basename(f))
        if mt and mt['case'] in CASE_LIST and int(mt['T']) >= 2:
            groups.setdefault((mt['dir'], float(mt['a2'])), []).append(
                (int(mt['T']), mt['case'], f))

    for (dir_name, a2), entries in sorted(groups.items()):
        print(f"\n############## dir = {dir_name}, a2 = {a2:.2f} ##############")
        print(f"{'T':>4} {'case':>4} " + ' '.join(f'{k:>11}' for k in VERSIONS)
              + f" {'den=0 (s)':>10} {'den=0 (a)':>10}")
        fi, avg, hdr = {}, {}, ''
        for T, case, f in sorted(entries):
            d = load(f)
            if d['L'] != L_expected:
                print(f"  [warning] {os.path.basename(f)} has L={d['L']}, "
                      f"expected L={L_expected} for folder '{results_dir}'")
            hdr = (rf"L={d['L']}, N={d['N']}, $\mu$={float(d['mu']):g}, "
                   rf"$V_s$={d['V_s']:g}, $a^2$={d['a2']:g}, dir={d['dir_name']}, "
                   rf"$\sigma^2$={d['sigma_e2']:g}, $\theta$={float(d['theta']):g}, "
                   f"generations={int(d['maxiter'])}, burn-in={d['BURN_IN']}, "
                   f"G every {int(d['TRAIT_REC_EVERY'])} gens, "
                   f"error bars: {N_BLOCKS}-block jackknife")

            G    = G_series(d)                                 # (n, T, T)
            diag = np.einsum('ktt->kt', G)                     # (n, T)  G_ii
            offs = {False: G.sum(axis=2) - diag,               # signed sum_{j!=i} G_ij
                    True:  np.abs(G).sum(axis=2) - np.abs(diag)}   # sum_{j!=i} |G_ij|
            fi[(case, T)], avg[(case, T)] = {}, {}
            for key, (use_abs, rom, _) in VERSIONS.items():
                off = offs[use_abs]
                fi[(case, T)][key]  = f_of_i(diag, off, rom)
                avg[(case, T)][key] = (trait_avg(diag, off, rom), block_se(diag, off, rom))
            print(f"{T:>4} {case:>4} "
                  + ' '.join(f'{avg[(case, T)][k][0]:>11.4g}' for k in VERSIONS)
                  + f" {np.mean(offs[False] == 0):>10.2%} {np.mean(offs[True] == 0):>10.2%}")

        T_plot = sorted({T for (_, T) in fi})
        cases  = [c for c in CASE_LIST if any((c, T) in fi for T in T_plot)]
        jobs.append((results_dir, fi, avg, T_plot, cases, dir_name, a2, hdr, out))

# One y-range for every abs panel in every figure of every folder, so the L = 100 and
# L = 1000 PDFs can be compared side by side.  It covers every f(i) and every
# trait average with its error bar.
vals = []
for _, fi, avg, *_ in jobs:
    for key in ('abs_mr', 'abs_rm'):
        for v in fi.values():
            vals.append(np.ravel(v[key]))
        for v in avg.values():
            m, se = v[key]
            se = se if np.isfinite(se) else 0.0
            vals.append(np.array([m - se, m, m + se]))
vals = np.concatenate(vals) if vals else np.array([])
vals = vals[np.isfinite(vals) & (vals > 0)]
abs_ylim = (vals.min() / 1.3, vals.max() * 1.3) if len(vals) else None
if abs_ylim:
    print(f"\nShared y-range of the abs panels: {abs_ylim[0]:.3g} .. {abs_ylim[1]:.3g}")

for results_dir, fi, avg, T_plot, cases, dir_name, a2, hdr, out in jobs:
    print(f"\n--- figures for {results_dir}, dir = {dir_name}, a2 = {a2:.2f} ---")
    figure_by_trait(fi, T_plot, cases, dir_name, a2, hdr, out, abs_ylim)
    figure_over_T(avg, T_plot, cases, dir_name, a2, hdr, out, abs_ylim)

print("\nDone.")
