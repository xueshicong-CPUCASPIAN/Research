# -*- coding: utf-8 -*-
r"""
Autocovariance of the per-locus selection kernel  w_i(t) = a_i . delta(t)  over lag tau:

    C(tau) = Cov( w_i(t), w_i(t - tau) )

PLOTTING ONLY -- this script runs no simulation.  It reads the cross_term_data_*.npz
files written by sweep_T_4cases_violin.py (tracked replicate TRACK_REP, recorded every
REC_EVERY generations, post burn-in only).

Why this quantity.  In p_prime_sel_opt the allele's log-odds change per generation is
(a_i . delta + |a_i|^2 (p - 1/2)) / V_s, so w_i / V_s is the DIRECTIONAL selection
coefficient on allele i, and it fluctuates because delta does.  C(tau) is the
autocovariance of that fluctuating selection coefficient.  Its height C(0) is how
strong the fluctuation is, and its width (the correlation time) says whether the
selection on an allele keeps pushing the same way for longer than the allele lives
(acts like a sweep) or flips sign quickly (averages out).

Estimator.  Pairs (t, t - tau) are pooled over loci and over time.  A pair is used only
if the locus is segregating at both times AND holds the SAME allele (same birth
generation `aid`), so an allele lost and replaced by a new mutation is never paired
with its successor.  C(tau) = mean(x y) - mean(x) mean(y) over those pairs.  Note that
requiring survival over tau conditions on long-lived alleles, which grows with tau.

Output (per (direction, a2); each layout as autocov = C(tau) and autocorr = C(tau)/C(0)):
  autocov_w_by_case_<dir>_a2_<a2>.pdf,  autocorr_w_by_case_<dir>_a2_<a2>.pdf
      -- 6 panels (cases A-F), one line per T
  autocov_w_by_T_<dir>_a2_<a2>.pdf,     autocorr_w_by_T_<dir>_a2_<a2>.pdf
      -- 5 panels (T values), one line per case, plus the sigma^2=0 baseline in black
"""

import os
import glob
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── output directory (must match sweep_T_4cases_violin.py) ────────────────────
RESULTS_DIR = 'results Sep 15'
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', RESULTS_DIR)
def out(name):  return os.path.join(OUTDIR, name)

# ── settings ─────────────────────────────────────────────────────────────────
TAU_MAX   = 5000    # largest lag, in generations
TAU_STEP  = 25      # lag spacing, in generations (rounded to a multiple of REC_EVERY)
MIN_PAIRS = 200     # lags with fewer valid pairs are left blank

CASE_LIST = ['A', 'B', 'C', 'D', 'E', 'F']
case_labels = {
    'A': r'A: $\Sigma_{ii}=\sigma^2,\ \Sigma_{ij}=+\sigma^2$',
    'B': r'B: $\Sigma_{ii}=\sigma^2,\ \Sigma_{ij}=-\sigma^2$',
    'C': r'C: $\Sigma_{ii}=\sigma^2/T,\ \Sigma_{ij}=+\sigma^2/T$',
    'D': r'D: $\Sigma_{ii}=\sigma^2/T,\ \Sigma_{ij}=-\sigma^2/T$',
    'E': r'E: $\Sigma_{ii}=\sigma^2/T,\ \Sigma_{ij}=0$',
    'F': r'F: $\Sigma_{ii}=\sigma^2,\ \Sigma_{ij}=0$',
}
case_colors = {'A': 'C0', 'B': 'C3', 'C': 'C2', 'D': 'C1', 'E': 'C4', 'F': 'C5'}

# Every figure is drawn twice: as the autocovariance C(tau), and as the autocorrelation
# C(tau)/C(0).  C(0) shrinks roughly like 1/T, so the covariance panels compare
# strengths while the correlation panels (all starting at 1) compare correlation times.
MODES = {
    'autocov':  dict(what='autocovariance',
                     ylabel=r'$\mathrm{Cov}(\vec a_i\cdot\vec\delta_t,\ '
                            r'\vec a_i\cdot\vec\delta_{t-\tau})$'),
    'autocorr': dict(what='autocorrelation',
                     ylabel=r'$\mathrm{Corr}(\vec a_i\cdot\vec\delta_t,\ '
                            r'\vec a_i\cdot\vec\delta_{t-\tau})$'),
}


def as_mode(C, mode):
    if mode == 'autocorr':
        return C / C[0] if np.isfinite(C[0]) and C[0] > 0 else np.full_like(C, np.nan)
    return C


def autocov(gen, p, w, aid, burn_in):
    """Lags (generations) and C(tau) for one tracked-replicate record."""
    m = gen >= burn_in
    gen = gen[m]
    w   = np.asarray(w[m], dtype=float)
    seg = np.asarray(p[m]) > 0
    aid = aid[m]
    dg  = int(gen[1] - gen[0])
    ks  = np.arange(0, TAU_MAX // dg + 1, max(1, TAU_STEP // dg))
    ks  = ks[ks < len(gen)]
    C   = np.full(len(ks), np.nan)
    n   = len(gen)
    for j, k in enumerate(ks):
        ok = seg[k:] & seg[:n - k] & (aid[k:] == aid[:n - k])
        if ok.sum() < MIN_PAIRS:
            continue
        x, y = w[k:][ok], w[:n - k][ok]
        C[j] = np.mean(x * y) - x.mean() * y.mean()
    return ks * dg, C


def style(ax, title, ylabel):
    ax.axhline(0, color='0.5', lw=0.6)
    ax.set_xlabel(r'lag $\tau$ (generations)')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)


# Colours are explained once, in a legend row under the suptitle, instead of in every
# panel.  Margins are set by hand: tight_layout also reserves room for the suptitle and
# would push the panels far below the legend.
def finish(fig, title, hdr, handles, ncol, top, fname):
    fig.suptitle(f'{title}\n{hdr}', fontsize=11, y=0.995)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.915),
               ncol=ncol, fontsize=9, frameon=False)
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.07, top=top,
                        hspace=0.35, wspace=0.28)
    fig.savefig(out(fname), bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {fname}")


def plot_by_case(acf, T_seen, cases, T_colors, mode, hdr, dir_name, a2):
    """One panel per case (A-F), one line per T."""
    what, ylabel = MODES[mode]['what'], MODES[mode]['ylabel']
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5), squeeze=False)
    for ax, case in zip(axes.ravel(), CASE_LIST):
        if case not in cases:
            ax.axis('off')
            continue
        for T in T_seen:
            if (case, T) in acf:
                tau, C = acf[(case, T)]
                ax.plot(tau, as_mode(C, mode), color=T_colors[T], lw=1.3)
        style(ax, case_labels[case], ylabel)
    handles = [plt.Line2D([], [], color=T_colors[T], lw=2, label=f'T = {T}')
               for T in T_seen]
    finish(fig, rf'{what} of $\vec a_i\cdot\vec\delta$ by case', hdr, handles,
           ncol=len(handles), top=0.85,
           fname=f'{mode}_w_by_case_{dir_name}_a2_{a2:.2f}.pdf')


def plot_by_T(acf, acf0, T_seen, cases, mode, hdr, dir_name, a2):
    """One panel per T, one line per case, plus the sigma^2=0 baseline."""
    what, ylabel = MODES[mode]['what'], MODES[mode]['ylabel']
    ncol = 3
    nrow = int(np.ceil(len(T_seen) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(16, 4.75 * nrow), squeeze=False)
    for ax, T in zip(axes.ravel(), T_seen):
        for case in cases:
            if (case, T) in acf:
                tau, C = acf[(case, T)]
                ax.plot(tau, as_mode(C, mode), color=case_colors[case], lw=1.3)
        tau, C = acf0[T]
        ax.plot(tau, as_mode(C, mode), color='k', ls='--', lw=1.2)
        style(ax, f'T = {T}', ylabel)
    for ax in axes.ravel()[len(T_seen):]:
        ax.axis('off')
    handles = ([plt.Line2D([], [], color=case_colors[c], lw=2, label=case_labels[c])
                for c in cases]
               + [plt.Line2D([], [], color='k', ls='--', lw=1.5,
                             label=r'$\sigma^2=0$ baseline')])
    finish(fig, rf'{what} of $\vec a_i\cdot\vec\delta$ by $T$', hdr, handles,
           ncol=4, top=0.82 if nrow > 1 else 0.70,
           fname=f'{mode}_w_by_T_{dir_name}_a2_{a2:.2f}.pdf')


# ── main ─────────────────────────────────────────────────────────────────────
PATTERN = re.compile(r'cross_term_data_(?P<dir>\w+?)_T(?P<T>\d+)_'
                     r'case(?P<case>[A-F])_a2_(?P<a2>[\d.]+)\.npz$')

files = sorted(glob.glob(out('cross_term_data_*.npz')))
if not files:
    raise SystemExit(f'No cross_term_data_*.npz in {OUTDIR}.\n'
                     'Run sweep_T_4cases_violin.py first.')

groups = {}
for f in files:
    mt = PATTERN.search(os.path.basename(f))
    if mt:
        groups.setdefault((mt['dir'], float(mt['a2'])), []).append(
            (int(mt['T']), mt['case'], f))

for (dir_name, a2), entries in sorted(groups.items()):
    print(f"\n############## dir = {dir_name}, a2 = {a2:.2f} ##############")
    acf, acf0, hdr = {}, {}, ''
    for T, case, f in sorted(entries):
        z = np.load(f)
        if 'aid' not in z.files:
            raise SystemExit(f'{f} has no allele-birth record (aid).\n'
                             'It was written by an older sweep; re-run sweep_T_4cases_violin.py.')
        burn = int(z['BURN_IN'])
        acf[(case, T)] = autocov(z['gen'], z['p'], z['w'], z['aid'], burn)
        if T not in acf0:                  # sigma^2=0 baseline, identical in every case file
            acf0[T] = autocov(z['gen0'], z['p0'], z['w0'], z['aid0'], burn)
        hdr = (rf"L={int(z['L'])}, N={int(z['N'])}, $\mu$={float(z['mu']):g}, "
               rf"$V_s$={float(z['V_s']):g}, $a^2$={a2:g}, A~{z['dist_name']}, "
               rf"dir={dir_name}, $\sigma^2$={float(z['sigma_e2']):g}, "
               rf"$\theta$={float(z['theta']):g}, generations={int(z['maxiter'])}, "
               f"burn-in={burn}\n"
               f"tracked replicate {int(z['TRACK_REP'])} of {int(z['rep'])}, "
               f"recorded every {int(z['REC_EVERY'])} gens;  "
               rf"$\tau_{{max}}$={TAU_MAX}, $\tau$ step={TAU_STEP} gens, "
               f"min pairs per lag={MIN_PAIRS}")
        print(f"  T={T:>3} case {case}: C(0) = {acf[(case, T)][1][0]:.4g}")

    T_seen    = sorted({T for (_, T) in acf})
    cases     = [c for c in CASE_LIST if any((c, T) in acf for T in T_seen)]
    T_colors  = {T: plt.cm.viridis(i / max(1, len(T_seen) - 1)) for i, T in enumerate(T_seen)}

    for mode in MODES:
        plot_by_case(acf, T_seen, cases, T_colors, mode, hdr, dir_name, a2)
        plot_by_T(acf, acf0, T_seen, cases, mode, hdr, dir_name, a2)

print("\nDone.")
