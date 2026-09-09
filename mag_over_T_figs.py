# -*- coding: utf-8 -*-
r"""
How |a_i|, ||delta|| and a_i . delta scale with the NUMBER OF TRAITS T.

PLOTTING ONLY -- this script runs no simulation.  Like cross_term_figs.py it reads the
cross_term_data_*.npz files written by sweep_T_4cases_violin.py.

The single-locus close-up of  a_i . delta  lives in cross_term_figs.py, as panels F and
G of fig1abc_cross_*.pdf.  This script answers the other half of the question: not "why
did this locus jump" but "how big is a_i . delta in the first place, and where does its
T-dependence come from".

Analytic predictions, drawn on top of the measurements.  With the model as coded in
sweep_T_4cases_violin.py,

    a_i = sqrt(A/T) * D,   A = a2 constant,   D_t ~ N(0,1) ('gauss') or +-1 ('pm')

  (i)   E|a_i|^2 = (a2/T) E||D||^2 = a2                    -- NO T-dependence at all.
        For 'pm' this is not just the mean: ||D||^2 = T identically, so
        |a_i| = sqrt(a2) exactly at every locus, with zero variance.  For 'gauss' the
        mean is fixed and the relative spread is sqrt(2/T), i.e. it TIGHTENS with T.
  (ii)  delta = opt - zbar has NO closed-form T-scaling here, and this panel is a
        measurement rather than a check of one.  theta = 0 in the parameter block, so
        the optimum is a pure RANDOM WALK with per-generation increment covariance
        Sigma -- it has no stationary distribution of its own.  What makes the LAG
        delta stationary is the selection response itself,
            d zbar_m / dt = (G delta)_m / V_s,
        so with G ~ diag(V_g) the lag balances at
            Var(delta_m) ~ Sigma_tt V_s / (2 V_g,m),
        and V_g,m = 2 sum_i a_{im}^2 p q carries its own factor a2/T from the effect
        scaling, while E[pq] responds in turn to the per-locus selection strength of
        (iv).  That feedback loop is exactly what the simulation is for; asserting a
        clean power law here would be guessing.  What IS safe to say: Sigma_tt enters
        the numerator, so cases A/B (Sigma_tt = sigma^2) must sit above cases C/D
        (Sigma_tt = sigma^2/T) and grow faster in T.  Only the DIAGONAL enters; the
        off-diagonal sign (A vs B, C vs D) changes the direction of delta, not its
        length.  The panel therefore reports a fitted log-log slope per case.
  (iii) conditional on delta, a_i . delta = sqrt(A/T) sum_t D_t delta_t has mean 0 and
        variance (a2/T)||delta||^2 -- identical for BOTH direction models, since only
        the first two moments of D_t enter.  Hence
            RMS(a_i . delta) = sqrt(a2/T) * RMS||delta||
        exactly, whatever ||delta|| turns out to do.  This is a genuine consistency
        check between two independently measured columns, not a fitted curve, which is
        why the dashed lines in panel (c) should land on the markers.
  (iv)  in the ratio, ||delta|| cancels outright:
            RMS( |a_i . delta| / ||delta|| ) = sqrt(a2 / T)  =:  f(T)
        independent of the case, of sigma^2, of the direction model, and -- crucially,
        given (ii) -- of whatever the optimum process happens to do.  Geometrically it
        is |a_i| cos(angle) with |a_i| = sqrt(a2) fixed and E[cos^2] = 1/T for a random
        direction in T dimensions: the only T-dependence left is dimensionality.  This
        is the one prediction here that is exact rather than a scaling argument.

Panel (d) is the test that matters.  The derivation of (iv) assumes a_i is independent
of delta, which is true at the moment the mutation is drawn but NOT for the loci that
survive: selection retains the ones aligned with delta.  Any systematic gap between the
measured curve and the black sqrt(a2/T) line is that conditioning, and is the
interesting part of the figure.

Output (one per (direction, a2)):
  mag_over_T_<dir>_a2_<a2>.pdf
"""

import os
import glob
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── output directory (must match sweep_T_4cases_violin.py / cross_term_figs.py) ──
RESULTS_DIR = 'results Aug 10'
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', RESULTS_DIR)
def out(name):  return os.path.join(OUTDIR, name)

CASE_LIST = ['A', 'B', 'C', 'D']
case_labels = {
    'A': r'A: $\Sigma_{ii}=\sigma^2,\ \Sigma_{ij}=+\sigma^2$',
    'B': r'B: $\Sigma_{ii}=\sigma^2,\ \Sigma_{ij}=-\sigma^2$',
    'C': r'C: $\Sigma_{ii}=\sigma^2/T,\ \Sigma_{ij}=+\sigma^2/T$',
    'D': r'D: $\Sigma_{ii}=\sigma^2/T,\ \Sigma_{ij}=-\sigma^2/T$',
}
case_colors = {'A': 'C0', 'B': 'C3', 'C': 'C2', 'D': 'C1', '0': 'k'}


def load(path):
    z = np.load(path)
    d = {k: z[k] for k in z.files}
    for k in ('T', 'L', 'N', 'BURN_IN', 'REC_EVERY', 'TRAIT_REC_EVERY', 'TRACK_REP',
              'rep'):
        if k in d:
            d[k] = int(d[k])
    for k in ('V_s', 'a2', 'sigma_e2'):
        if k in d:
            d[k] = float(d[k])
    for k in ('dir_name', 'case'):
        if k in d:
            d[k] = str(d[k])
    return d


def param_str(d):
    return (f"L={d['L']}, N={d['N']}, $V_s$={d['V_s']:g}, $a^2$={d['a2']:g}, "
            rf"$\sigma^2$={d['sigma_e2']:g}, dir={d['dir_name']}, "
            f"replicate {d['TRACK_REP']} of {d['rep']}")


def effect_norms(d):
    """|a_i| for the segregating loci, recovered from the recorded a_{im} delta_m.

    The simulation never stores the effect matrix itself, but it does store
    ad[k, i, m] = a_{im} * delta_m at the trait-recording generations, and delta at
    every recording generation.  Dividing them back out is exact up to the float32
    rounding of `ad`, and that rounding is RELATIVE, so the recovered a_{im} stays
    accurate even for a trait whose delta_m happens to be small.
    """
    if 'ad' not in d or 'tgen' not in d:
        return np.array([])
    gen, tgen = d['gen'], d['tgen']
    idx = np.searchsorted(gen, tgen)
    ok = (idx < len(gen)) & (gen[np.clip(idx, 0, len(gen) - 1)] == tgen)
    if not ok.any():
        return np.array([])
    dlt = np.asarray(d['delta'], dtype=float)[idx[ok]]        # (n_ok, T)
    ad  = np.asarray(d['ad'],    dtype=float)[ok]             # (n_ok, L, T)
    seg = np.asarray(d['tp'], dtype=float)[ok] > 0            # (n_ok, L)
    with np.errstate(divide='ignore', invalid='ignore'):
        a = ad / dlt[:, None, :]
    a[~np.isfinite(a)] = np.nan                               # a trait with delta_m = 0
    return np.sqrt(np.nansum(a ** 2, axis=2))[seg]


def mag_over_T(mag, T_list, dir_name, a2, hdr):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.6))
    Tarr = np.array(T_list, dtype=float)

    def curve(ax, key, ylabel, title):
        for case in CASE_LIST:
            ax.plot(T_list, [mag[(case, T)][key] for T in T_list], marker='o',
                    color=case_colors[case], label=case_labels[case])
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xticks(T_list); ax.set_xticklabels([str(t) for t in T_list])
        ax.set_xlabel('Number of traits $T$'); ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3); ax.set_title(title, fontsize=10)

    # (a) |a_i| -- the prediction is a horizontal line, which is the whole point
    curve(axes[0], 'a', r'RMS $|\vec a_i|$ (segregating loci)',
          '(a) effect-vector length vs $T$')
    axes[0].plot(Tarr, np.full_like(Tarr, np.sqrt(a2)), color='k', ls='--', lw=1.0,
                 label=rf'analytic $\sqrt{{a^2}}={np.sqrt(a2):.3g}$, flat in $T$')
    axes[0].legend(fontsize=7)

    # (b) ||delta|| -- measured, not predicted (see (ii) in the docstring).  theta = 0
    # makes the optimum a random walk, so the lag is set by the selection response and
    # by V_g, which carries its own 1/T; a fitted exponent is the honest summary.
    curve(axes[1], 'delta', r'RMS $\|\vec\delta\|$',
          '(b) optimum displacement vs $T$\n'
          r'measured; legend gives the fitted $\|\vec\delta\|\propto T^{\,q}$')
    handles = []
    for case in CASE_LIST:
        ys = np.array([mag[(case, T)]['delta'] for T in T_list], dtype=float)
        ok = np.isfinite(ys) & (ys > 0)
        q = (np.polyfit(np.log(Tarr[ok]), np.log(ys[ok]), 1)[0] if ok.sum() > 1
             else np.nan)
        handles.append(plt.Line2D([], [], color=case_colors[case], marker='o',
                                  label=f'{case}:  $q = {q:+.2f}$'))
    axes[1].legend(handles=handles, fontsize=7, title='fitted slope',
                   title_fontsize=7)

    # (c) the inner product itself = (a) x (b), so it inherits the case dependence
    curve(axes[2], 'w', r'RMS $|\vec a_i\cdot\vec\delta|$',
          '(c) inner product vs $T$\n'
          r'dashed: exact identity $\sqrt{a^2/T}\,\mathrm{RMS}\|\vec\delta\|$'
          '\n(should land on the markers)')
    for case in CASE_LIST:
        axes[2].plot(T_list, [np.sqrt(a2 / T) * mag[(case, T)]['delta'] for T in T_list],
                     color=case_colors[case], ls='--', lw=0.9, alpha=0.7)
    axes[2].legend(fontsize=7)

    # (d) the ratio -- ||delta|| cancels, leaving f(T) = sqrt(a2/T) for every case
    curve(axes[3], 'ratio', r'RMS $|\vec a_i\cdot\vec\delta|\,/\,\|\vec\delta\|$',
          '(d) ratio vs $T$: dimensionality only\n'
          r'gap from the black line = selection retaining $\vec a_i$ aligned with $\vec\delta$')
    axes[3].plot(Tarr, np.sqrt(a2 / Tarr), color='k', ls='--', lw=1.2,
                 label=r'analytic $f(T)=\sqrt{a^2/T}$')
    axes[3].legend(fontsize=7)

    fig.suptitle(r'Magnitudes of $\vec a_i$, $\vec\delta$ and $\vec a_i\cdot\vec\delta$'
                 f' vs NUMBER OF TRAITS  ({hdr}; segregating loci, post burn-in)',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fname = f'mag_over_T_{dir_name}_a2_{a2:.2f}.pdf'
    fig.savefig(out(fname), bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved {fname}")


# ── main ──────────────────────────────────────────────────────────────────────
PATTERN = re.compile(r'cross_term_data_(?P<dir>\w+?)_T(?P<T>\d+)_'
                     r'case(?P<case>[A-D])_a2_(?P<a2>[\d.]+)\.npz$')

files = sorted(glob.glob(out('cross_term_data_*.npz')))
if not files:
    raise SystemExit(f'No cross_term_data_*.npz in {OUTDIR}.\n'
                     'Run sweep_T_4cases_violin.py first -- it is the only script '
                     'that simulates.')

groups = {}
for f in files:
    mt = PATTERN.search(os.path.basename(f))
    if mt:
        groups.setdefault((mt['dir'], float(mt['a2'])), []).append((int(mt['T']),
                                                                    mt['case'], f))

for (dir_name, a2), entries in sorted(groups.items()):
    print(f"\n############## dir = {dir_name}, a2 = {a2:.2f} ##############")
    mag, hdr = {}, ''
    T_seen = sorted({T for T, _, _ in entries})

    for T, case, f in sorted(entries):
        d = load(f)
        hdr = param_str(d)
        m = d['gen'] >= d['BURN_IN']
        W = np.asarray(d['w'], dtype=float)[m]
        P = np.asarray(d['p'], dtype=float)[m]
        dn = np.linalg.norm(np.asarray(d['delta'], dtype=float)[m], axis=1)
        seg = P > 0                                   # stale rows are not the dynamics
        rat = W / dn[:, None]
        an = effect_norms(d)
        mag[(case, T)] = dict(
            a     = float(np.sqrt(np.nanmean(an ** 2))) if an.size else np.nan,
            delta = float(np.sqrt(np.mean(dn ** 2))),
            w     = float(np.sqrt(np.mean(W[seg] ** 2))),
            ratio = float(np.sqrt(np.mean(rat[seg] ** 2))),
        )
        mm = mag[(case, T)]
        print(f"  T={T:>3} case {case}: RMS |a_i| = {mm['a']:.4g} "
              f"(analytic {np.sqrt(a2):.4g}), RMS ||delta|| = {mm['delta']:.4g}, "
              f"RMS |a.delta| = {mm['w']:.4g}, ratio = {mm['ratio']:.4g} "
              f"(analytic {np.sqrt(a2 / T):.4g})")

    missing = [(c, T) for T in T_seen for c in CASE_LIST if (c, T) not in mag]
    if missing:
        print(f"  Skipping mag_over_T: missing {len(missing)} (case, T) combinations, "
              f"e.g. {missing[:3]}")
        continue
    mag_over_T(mag, T_seen, dir_name, a2, hdr)

print("\nDone.")
