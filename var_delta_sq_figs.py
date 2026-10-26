# -*- coding: utf-8 -*-
r"""
Test of the hand derivation of Var(||delta||^2) against the simulation, for the two
DIAGONAL covariance cases only:

    E:  Sigma = (sigma^2 / T) I          F:  Sigma = sigma^2 I

PLOTTING ONLY -- this script runs no simulation.  It reads the cross_term_data_*.npz
files written by sweep_T_4cases_violin.py (tracked replicate, post burn-in).

The derivation (tau = time, t = trait, s = Sigma_tt):

  1. dz*_t = sqrt(s) dW_t, and Eq. (20) of the paper gives
         dzbar_t/dtau = (1/V_s) [ G_tt delta_t + sum_{j != t} G_tj delta_j ].
  2. Drop the cross term G_tj (E[G_tj] = 0).  Then each delta_t is an independent
     OU process,  d delta_t = -(G_tt / V_s) delta_t dtau + sqrt(s) dW_t.
  3. Stationary:  delta_t ~ N(0, v_t),   v_t = s V_s / (2 G_tt),   G_tt = V_g,t.
  4. Independence + Gaussianity  (E[X^4] = 3 v^2):
         E||delta||^2   = sum_t v_t
         Var(||delta||^2) = 2 sum_t v_t^2
     and, if every v_t = v, the T-free check  T Var / E^2 = 2.

The predictions below use the MEASURED per-trait V_g,t (time-averaged), so they do
NOT assume V_G is independent of T.  With V_g,t = V_G / T they reduce to
v = s V_s T / (2 V_G), i.e. Var ~ T^3 (case F) and ~ T (case E) when V_G is flat.

Panels:
  (a) E||delta||^2       measured vs prediction 3-4
  (b) Var(||delta||^2)   measured vs prediction 3-4, fitted log-log slope
  (c) T Var / (E)^2      measured vs 2
  (d) total V_G = sum_t V_g,t vs T  -- assumption "V_G independent of T"
  (e) effective number of segregating loci n_eff vs T  -- step 2 needs T << n_eff

Error bars: block jackknife over N_BLOCKS consecutive blocks of the post-burn-in
series.  delta is autocorrelated over ~V_s T / V_G generations, which at T = 100 is a
few thousand, so there are only ~10 independent samples there; the bars say so.

Output:  var_delta_sq_<dir>_a2_<a2>.pdf   (in the results folder)
"""

import os
import glob
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── output directory (must match sweep_T_4cases_violin.py / mag_over_T_figs.py) ──
RESULTS_DIR = 'results Sep 15'
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', RESULTS_DIR)
def out(name):  return os.path.join(OUTDIR, name)

CASE_LIST = ['E', 'F']                      # diagonal Sigma only
case_labels = {
    'E': r'E: $\Sigma=(\sigma^2/T)\,I$',
    'F': r'F: $\Sigma=\sigma^2 I$',
}
case_colors = {'E': 'C4', 'F': 'C5'}
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


def sigma_tt(d):
    """Diagonal entry s = Sigma_tt of the optimum-increment covariance."""
    return d['sigma_e2'] / d['T'] if d['case'] == 'E' else d['sigma_e2']


def stats(n2, Vg, s, V_s):
    """Measured and predicted moments from one post-burn-in series.

    n2: (n,) ||delta||^2 per recording;  Vg: (n, T) per-trait V_g,t per recording.
    """
    T = Vg.shape[1]
    v = s * V_s / (2 * Vg.mean(axis=0))                 # (T,) predicted Var(delta_t)
    E, V = n2.mean(), n2.var()
    return dict(E=E, Var=V, ratio=T * V / E ** 2,
                E_pred=v.sum(), Var_pred=2 * (v ** 2).sum(),
                ratio_pred=T * 2 * (v ** 2).sum() / v.sum() ** 2,   # = 2 if all v_t equal
                VG=Vg.sum(axis=1).mean())


def jackknife(n2, Vg, s, V_s):
    """Block-jackknife standard error of every entry of stats()."""
    blocks = np.array_split(np.arange(len(n2)), N_BLOCKS)
    reps = []
    for b in blocks:
        keep = np.setdiff1d(np.arange(len(n2)), b)
        reps.append(stats(n2[keep], Vg[keep], s, V_s))
    B = len(reps)
    return {k: float(np.sqrt((B - 1) / B * np.sum(
                (np.array([r[k] for r in reps]) - np.mean([r[k] for r in reps])) ** 2)))
            for k in reps[0]}


def n_eff(P):
    """Effective number of segregating loci, (sum pq)^2 / sum (pq)^2, per recording."""
    pq = P * (1 - P)
    return (pq.sum(axis=1) ** 2 / np.maximum((pq ** 2).sum(axis=1), 1e-300)).mean()


def slope(T_list, ys):
    ys = np.asarray(ys, dtype=float)
    ok = np.isfinite(ys) & (ys > 0)
    return (np.polyfit(np.log(np.asarray(T_list, float)[ok]), np.log(ys[ok]), 1)[0]
            if ok.sum() > 1 else np.nan)


def figure(res, T_list, cases, dir_name, a2, hdr):
    fig, axes = plt.subplots(1, 5, figsize=(25, 6.0))
    Tarr = np.array(T_list, dtype=float)

    def setup(ax, ylabel, title, logy=True):
        ax.set_xscale('log')
        if logy:
            ax.set_yscale('log')
        ax.set_xticks(T_list); ax.set_xticklabels([str(t) for t in T_list])
        ax.set_xlabel('Number of traits $T$'); ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3); ax.set_title(title, fontsize=10)

    def measured_vs_pred(ax, key):
        handles = []
        for c in cases:
            m  = [res[(c, T)][key] for T in T_list]
            se = [res[(c, T)]['se'][key] for T in T_list]
            p  = [res[(c, T)][key + '_pred'] for T in T_list]
            ax.errorbar(T_list, m, yerr=se, marker='o', color=case_colors[c],
                        capsize=3, lw=1.5)
            ax.plot(T_list, p, color=case_colors[c], ls='--', lw=1.2)
            handles.append(plt.Line2D([], [], color=case_colors[c], lw=2,
                           label=f'{c}: slope meas {slope(T_list, m):+.2f}, '
                                 f'pred {slope(T_list, p):+.2f}'))
        handles += [plt.Line2D([], [], color='k', marker='o', lw=1.5, label='measured'),
                    plt.Line2D([], [], color='k', ls='--', lw=1.2, label='prediction')]
        ax.legend(handles=handles, fontsize=7)

    # (a) mean
    setup(axes[0], r'$E\,\|\vec\delta\|^2$',
          r'(a) mean: $E\|\vec\delta\|^2$ vs $\sum_t v_t$' '\n'
          r'$v_t=\Sigma_{tt}V_s/(2V_{g,t})$ with measured $V_{g,t}$')
    measured_vs_pred(axes[0], 'E')

    # (b) variance, with T and T^3 guides anchored at T = 1 of case F
    setup(axes[1], r'Var$(\|\vec\delta\|^2)$',
          r'(b) variance: Var$\|\vec\delta\|^2$ vs $2\sum_t v_t^2$' '\n'
          r'grey guides: $\propto T$ and $\propto T^3$')
    measured_vs_pred(axes[1], 'Var')
    y0 = res[(cases[-1], T_list[0])]['Var_pred']
    for q, ls in ((1, ':'), (3, '-.')):
        axes[1].plot(Tarr, y0 * (Tarr / Tarr[0]) ** q, color='grey', ls=ls, lw=1.0)

    # (c) normalised ratio, prediction 2
    setup(axes[2], r'$T\,$Var$(\|\vec\delta\|^2)\,/\,(E\|\vec\delta\|^2)^2$',
          '(c) T-free check\n'
          r'independent Gaussian $\delta_t$ with equal $v_t$ $\Rightarrow$ exactly 2',
          logy=False)
    for c in cases:
        axes[2].errorbar(T_list, [res[(c, T)]['ratio'] for T in T_list],
                         yerr=[res[(c, T)]['se']['ratio'] for T in T_list],
                         marker='o', color=case_colors[c], capsize=3, lw=1.5)
    axes[2].axhline(2, color='k', ls='--', lw=1.2, label='prediction = 2')
    axes[2].legend(fontsize=7)

    # (d) total V_G
    setup(axes[3], r'$V_G=\sum_t V_{g,t}$ (time mean)',
          '(d) total genetic variance vs $T$\n'
          r'assumption: $V_G$ independent of $T$ (flat)')
    for c in cases:
        ys = [res[(c, T)]['VG'] for T in T_list]
        axes[3].plot(T_list, ys, marker='o', color=case_colors[c],
                     label=f'{c}: slope {slope(T_list, ys):+.2f}')
    axes[3].legend(fontsize=7)

    # (e) n_eff vs T
    setup(axes[4], r'$n_{\rm eff}=(\sum pq)^2/\sum (pq)^2$',
          '(e) effective number of segregating loci\n'
          r'dropping $G_{tj}$ needs $T\ll n_{\rm eff}$ (below the black line fails)')
    for c in cases:
        axes[4].plot(T_list, [res[(c, T)]['neff'] for T in T_list], marker='o',
                     color=case_colors[c])
    axes[4].plot(Tarr, Tarr, color='k', ls='--', lw=1.0, label=r'$n_{\rm eff}=T$')
    axes[4].legend(fontsize=7)

    fig.suptitle(r'Var$(\|\vec\delta\|^2)$ vs NUMBER OF TRAITS: hand derivation vs '
                 'simulation (diagonal $\\Sigma$ only, tracked replicate, post burn-in)'
                 f'\n{hdr}', fontsize=11, y=1.00)
    fig.legend(handles=[plt.Line2D([], [], color=case_colors[c], marker='o',
                                   label=case_labels[c]) for c in cases],
               loc='upper center', bbox_to_anchor=(0.5, 0.90), ncol=len(cases),
               fontsize=9, frameon=False)
    fig.subplots_adjust(left=0.04, right=0.99, bottom=0.10, top=0.74, wspace=0.30)
    fname = f'var_delta_sq_{dir_name}_a2_{a2:.2f}.pdf'
    fig.savefig(out(fname), bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved {fname}")


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
    print(f"{'T':>4} {'case':>4} {'E|d|^2':>10} {'pred':>10} {'Var|d|^2':>11} "
          f"{'pred':>11} {'T*Var/E^2':>10} {'V_G':>7} {'n_eff':>6} {'tau_a':>7}")
    res, hdr = {}, ''
    T_seen = sorted({T for T, _, _ in entries})

    for T, case, f in sorted(entries):
        d = load(f)
        hdr = (rf"L={d['L']}, N={d['N']}, $\mu$={float(d['mu']):g}, "
               rf"$V_s$={d['V_s']:g}, $a^2$={d['a2']:g}, dir={d['dir_name']}, "
               rf"$\sigma^2$={d['sigma_e2']:g}, $\theta$={float(d['theta']):g}, "
               f"generations={int(d['maxiter'])}, burn-in={d['BURN_IN']}, "
               f"recorded every {d['REC_EVERY']} gens, "
               f"error bars: {N_BLOCKS}-block jackknife")
        m   = d['gen'] >= d['BURN_IN']
        dl  = np.asarray(d['delta'], dtype=float)[m]          # (n, T)
        Vg  = np.asarray(d['Vg'],    dtype=float)[m]          # (n, T)  G_tt = V_g,t
        P   = np.asarray(d['p'],     dtype=float)[m]          # (n, L)
        n2  = (dl ** 2).sum(axis=1)                           # ||delta||^2
        s   = sigma_tt(d)

        r = stats(n2, Vg, s, d['V_s'])
        r['se']   = jackknife(n2, Vg, s, d['V_s'])
        r['neff'] = n_eff(P)
        tau_a     = d['V_s'] * T / r['VG']                    # relaxation time of delta_t
        res[(case, T)] = r
        print(f"{T:>4} {case:>4} {r['E']:>10.4g} {r['E_pred']:>10.4g} "
              f"{r['Var']:>11.4g} {r['Var_pred']:>11.4g} {r['ratio']:>10.3g} "
              f"{r['VG']:>7.4f} {r['neff']:>6.1f} {tau_a:>7.0f}")

    cases = [c for c in CASE_LIST if all((c, T) in res for T in T_seen)]
    for c in CASE_LIST:
        if c not in cases:
            print(f"  [note] case {c} not available at every T; omitted")
    if cases:
        figure(res, T_seen, cases, dir_name, a2, hdr)

print("\nDone.")
