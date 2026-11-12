# -*- coding: utf-8 -*-
r"""
Autocovariance of the optimum displacement delta itself, trait by trait, and the raw
trace of delta_1(t), over the number of traits T.

PLOTTING ONLY -- this script runs no simulation.  It reads the cross_term_data_*.npz
files written by sweep_T_4cases_violin.py (tracked replicate TRACK_REP, delta recorded
every REC_EVERY generations; only post burn-in generations are used).

Why.  autocov_w_figs.py shows that the autocovariance of the selection kernel
w_i = a_i . delta changes with T.  Writing w_i = sum_m a_{im} delta_m and taking the
effect components i.i.d. with mean zero (E[a_{im} a_{il}] = 0 for m != l),

    Cov( w_i(t), w_i(t - tau) )  ~  E[a_{im}^2] * sum_m C_m(tau)
                                  =  |a|^2 * (1/T) sum_m C_m(tau),

    C_m(tau) = Cov( delta_m(t), delta_m(t - tau) ).

So the T-dependence of the a_i . delta curves should already be visible in delta alone,
with the allele effects (and the survival conditioning on alleles) taken out.  Here:

  (1) C_m(tau) for every trait m;
  (2) the trait average  (1/T) sum_m C_m(tau);
  (3) the raw trace delta_1(t) for every T, to see what is going on dynamically.

Estimator.  C_m(tau) = mean(x y) - mean(x) mean(y) over all post burn-in pairs
(t, t - tau) of trait m.  Unlike autocov_w_figs.py there is no allele matching, so
every pair is used.

Output (per (direction, a2); (1) and (2) each as autocov = C(tau) and
autocorr = C(tau)/C(0)):
  autocov_delta_traits_<dir>_a2_<a2>.pdf,  autocorr_delta_traits_<dir>_a2_<a2>.pdf
      -- rows = cases, cols = T.  Grey: every trait m; colour: trait 1; black: the
         trait average.  (In the autocorr file each curve is divided by its own C(0).)
  autocov_delta_mean_by_case_<dir>_a2_<a2>.pdf,  autocorr_delta_mean_by_case_...pdf
      -- trait average; 6 panels (cases A-F), one line per T
  autocov_delta_mean_by_T_<dir>_a2_<a2>.pdf,     autocorr_delta_mean_by_T_...pdf
      -- trait average; one panel per T, one line per case, plus the sigma^2=0
         baseline in black
  delta_trace_<dir>_a2_<a2>.pdf
      -- delta_1(t) against generation; rows = cases, cols = T, one y-axis shared by
         every panel (TRACE_SHAREY); sigma^2=0 baseline in grey
"""

import os
import sys
import glob
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── output directory (must match sweep_T_4cases_violin.py) ────────────────────
# The folder can be given on the command line, e.g.
#     python delta_autocov_figs.py "results sep 15 OLD"      (the L = 100 data)
# With no argument the L = 1000 folder is used.
RESULTS_DIR = sys.argv[1] if len(sys.argv) > 1 else 'results_Nov_2_L_1000'
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', RESULTS_DIR)
def out(name):  return os.path.join(OUTDIR, name)

# ── settings ─────────────────────────────────────────────────────────────────
TAU_MAX   = 5000    # largest lag, in generations
TAU_STEP  = 25      # lag spacing, in generations (rounded to a multiple of REC_EVERY)

TRACE_START = None  # first generation of the trace window; None = start at burn-in
TRACE_LEN   = 2000  # length of the trace window, in generations
TRACE_TRAIT = 0     # which trait is traced (0 = trait 1, the focal trait)
TRACE_SHAREY = True # True = one y-axis for every panel; 'row' = shared within a case only

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

# Autocovariance compares strengths; autocorrelation (all curves start at 1) compares
# correlation times.
MODES = {
    'autocov':  dict(what='autocovariance',
                     y_m=r'$\mathrm{Cov}(\delta_{m,t},\ \delta_{m,t-\tau})$',
                     y_avg=r'$\frac{1}{T}\sum_m \mathrm{Cov}(\delta_{m,t},\ \delta_{m,t-\tau})$'),
    'autocorr': dict(what='autocorrelation',
                     y_m=r'$\mathrm{Corr}(\delta_{m,t},\ \delta_{m,t-\tau})$',
                     y_avg=r'$\sum_m C_m(\tau)\ /\ \sum_m C_m(0)$'),
}


def as_mode(C, mode):
    """C is (n_lag,) or (n_lag, T); each column is divided by its own lag-0 value."""
    if mode != 'autocorr':
        return C
    C0 = np.where(C[0] > 0, C[0], np.nan)
    return C / C0


def autocov_delta(gen, delta, burn_in):
    """Lags (generations) and C_m(tau), shape (n_lag, T), for one tracked record."""
    gen = np.asarray(gen)
    m   = gen >= burn_in
    gen = gen[m]
    d   = np.asarray(delta, dtype=float).reshape(len(m), -1)[m]     # (n, T)
    n   = len(gen)
    dg  = int(gen[1] - gen[0])
    ks  = np.arange(0, TAU_MAX // dg + 1, max(1, TAU_STEP // dg))
    ks  = ks[ks < n // 2]               # keep at least n/2 pairs at the largest lag
    C   = np.empty((len(ks), d.shape[1]))
    for j, k in enumerate(ks):
        x, y = d[k:], d[:n - k]
        C[j] = (x * y).mean(axis=0) - x.mean(axis=0) * y.mean(axis=0)
    return ks * dg, C


def trace(gen, delta, burn_in):
    """(generation, delta of trait TRACE_TRAIT) inside the trace window."""
    gen = np.asarray(gen)
    d   = np.asarray(delta, dtype=float).reshape(len(gen), -1)[:, TRACE_TRAIT]
    g0  = burn_in if TRACE_START is None else TRACE_START
    m   = (gen >= g0) & (gen < g0 + TRACE_LEN)
    return gen[m], d[m]


def style(ax, title, ylabel):
    ax.axhline(0, color='0.5', lw=0.6)
    ax.set_xlabel(r'lag $\tau$ (generations)')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)


# Colours are explained once, in a legend row under the suptitle.  Margins are set by
# hand: tight_layout also reserves room for the suptitle and would push the panels far
# below the legend.
def finish(fig, title, hdr, handles, ncol, top, fname, left=0.06):
    fig.suptitle(f'{title}\n{hdr}', fontsize=11, y=0.995)
    # legend top sits 0.8 inch below the figure top, whatever the figure height
    fig.legend(handles=handles, loc='upper center',
               bbox_to_anchor=(0.5, 1 - 0.8 / fig.get_figheight()),
               ncol=ncol, fontsize=9, frameon=False)
    fig.subplots_adjust(left=left, right=0.99, bottom=0.07, top=top,
                        hspace=0.35, wspace=0.28)
    fig.savefig(out(fname), bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {fname}")


def plot_traits(acf, T_seen, cases, mode, hdr, dir_name, a2):
    """(1) rows = cases, cols = T: every trait in grey, trait 1 in colour, average in black."""
    what, ylabel = MODES[mode]['what'], MODES[mode]['y_m']
    nr, nc = len(cases), len(T_seen)
    fig, axes = plt.subplots(nr, nc, figsize=(3.7 * nc, 2.9 * nr), squeeze=False)
    for r, case in enumerate(cases):
        for c, T in enumerate(T_seen):
            ax = axes[r, c]
            if (case, T) not in acf:
                ax.axis('off')
                continue
            tau, C = acf[(case, T)]
            ax.plot(tau, as_mode(C, mode), color='0.75', lw=0.5, alpha=0.6)
            ax.plot(tau, as_mode(C.mean(axis=1), mode), color='k', lw=1.5)
            ax.plot(tau, as_mode(C[:, 0], mode), color=case_colors[case], lw=1.1)
            style(ax, f'case {case},  T = {T}', ylabel if c == 0 else '')
            if r < nr - 1:
                ax.set_xlabel('')
    handles = [plt.Line2D([], [], color='0.6', lw=1, label='each trait $m$'),
               plt.Line2D([], [], color='C0', lw=2, label='trait 1 (case colour)'),
               plt.Line2D([], [], color='k', lw=2, label=r'trait average $\frac{1}{T}\sum_m$')]
    finish(fig, rf'{what} of $\delta_m$, trait by trait', hdr, handles, ncol=3,
           top=1 - 1.45 / (2.9 * nr),
           fname=f'{mode}_delta_traits_{dir_name}_a2_{a2:.2f}.pdf')


def plot_mean_by_case(acf, T_seen, cases, T_colors, mode, hdr, dir_name, a2):
    """(2) one panel per case (A-F), one line per T: the trait average."""
    what, ylabel = MODES[mode]['what'], MODES[mode]['y_avg']
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5), squeeze=False)
    for ax, case in zip(axes.ravel(), CASE_LIST):
        if case not in cases:
            ax.axis('off')
            continue
        for T in T_seen:
            if (case, T) in acf:
                tau, C = acf[(case, T)]
                ax.plot(tau, as_mode(C.mean(axis=1), mode), color=T_colors[T], lw=1.3)
        style(ax, case_labels[case], ylabel)
    handles = [plt.Line2D([], [], color=T_colors[T], lw=2, label=f'T = {T}')
               for T in T_seen]
    finish(fig, rf'trait-averaged {what} of $\delta$ by case', hdr, handles,
           ncol=len(handles), top=0.85,
           fname=f'{mode}_delta_mean_by_case_{dir_name}_a2_{a2:.2f}.pdf')


def plot_mean_by_T(acf, acf0, T_seen, cases, mode, hdr, dir_name, a2):
    """(2) one panel per T, one line per case, plus the sigma^2=0 baseline."""
    what, ylabel = MODES[mode]['what'], MODES[mode]['y_avg']
    ncol = 3
    nrow = int(np.ceil(len(T_seen) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(16, 4.75 * nrow), squeeze=False)
    for ax, T in zip(axes.ravel(), T_seen):
        for case in cases:
            if (case, T) in acf:
                tau, C = acf[(case, T)]
                ax.plot(tau, as_mode(C.mean(axis=1), mode), color=case_colors[case], lw=1.3)
        tau, C = acf0[T]
        ax.plot(tau, as_mode(C.mean(axis=1), mode), color='k', ls='--', lw=1.2)
        style(ax, f'T = {T}', ylabel)
    for ax in axes.ravel()[len(T_seen):]:
        ax.axis('off')
    handles = ([plt.Line2D([], [], color=case_colors[c], lw=2, label=case_labels[c])
                for c in cases]
               + [plt.Line2D([], [], color='k', ls='--', lw=1.5,
                             label=r'$\sigma^2=0$ baseline')])
    finish(fig, rf'trait-averaged {what} of $\delta$ by $T$', hdr, handles,
           ncol=4, top=0.82 if nrow > 1 else 0.70,
           fname=f'{mode}_delta_mean_by_T_{dir_name}_a2_{a2:.2f}.pdf')


def plot_trace(tr, tr0, T_seen, cases, hdr, dir_name, a2):
    """(3) delta_1(t): rows = cases, cols = T, y-axis shared as set by TRACE_SHAREY."""
    nr, nc = len(cases), len(T_seen)
    fig, axes = plt.subplots(nr, nc, figsize=(3.7 * nc, 2.3 * nr), squeeze=False,
                             sharex=True, sharey=TRACE_SHAREY)
    for r, case in enumerate(cases):
        for c, T in enumerate(T_seen):
            ax = axes[r, c]
            ax.axhline(0, color='0.5', lw=0.6)
            ax.grid(alpha=0.3)
            if T in tr0:
                ax.plot(*tr0[T], color='0.6', lw=0.7)
            if (case, T) in tr:
                ax.plot(*tr[(case, T)], color=case_colors[case], lw=0.9)
            ax.set_title(f'case {case},  T = {T}', fontsize=10)
            if c == 0:
                ax.set_ylabel(rf'$\delta_{{{TRACE_TRAIT + 1}}}(t)$')
            if r == nr - 1:
                ax.set_xlabel('generation')
    handles = ([plt.Line2D([], [], color=case_colors[c], lw=2, label=case_labels[c])
                for c in cases]
               + [plt.Line2D([], [], color='0.6', lw=1.5, label=r'$\sigma^2=0$ baseline')])
    finish(fig, rf'trace of $\delta_{{{TRACE_TRAIT + 1}}}(t)$ '
                f'({TRACE_LEN} generations; y-axis shared '
                f'{"by all panels" if TRACE_SHAREY is True else "along each row"})',
           hdr, handles, ncol=4, top=1 - 1.75 / (2.3 * nr),
           fname=f'delta_trace_{dir_name}_a2_{a2:.2f}.pdf')


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
    acf, acf0, tr, tr0, hdr = {}, {}, {}, {}, ''
    for T, case, f in sorted(entries):
        z = np.load(f)
        burn = int(z['BURN_IN'])
        acf[(case, T)] = autocov_delta(z['gen'], z['delta'], burn)
        tr[(case, T)]  = trace(z['gen'], z['delta'], burn)
        if T not in acf0:                  # sigma^2=0 baseline, identical in every case file
            acf0[T] = autocov_delta(z['gen0'], z['delta0'], burn)
            tr0[T]  = trace(z['gen0'], z['delta0'], burn)
        hdr = (rf"L={int(z['L'])}, N={int(z['N'])}, $\mu$={float(z['mu']):g}, "
               rf"$V_s$={float(z['V_s']):g}, $a^2$={a2:g}, A~{z['dist_name']}, "
               rf"dir={dir_name}, $\sigma^2$={float(z['sigma_e2']):g}, "
               rf"$\theta$={float(z['theta']):g}, generations={int(z['maxiter'])}, "
               f"burn-in={burn}\n"
               f"tracked replicate {int(z['TRACK_REP'])} of {int(z['rep'])}, "
               f"recorded every {int(z['REC_EVERY'])} gens;  "
               rf"$\tau_{{max}}$={TAU_MAX}, $\tau$ step={TAU_STEP} gens")
        C = acf[(case, T)][1]
        print(f"  T={T:>3} case {case}: C_1(0) = {C[0, 0]:.4g},  "
              f"(1/T) sum_m C_m(0) = {C[0].mean():.4g}")

    T_seen   = sorted({T for (_, T) in acf})
    cases    = [c for c in CASE_LIST if any((c, T) in acf for T in T_seen)]
    T_colors = {T: plt.cm.viridis(i / max(1, len(T_seen) - 1)) for i, T in enumerate(T_seen)}

    for mode in MODES:
        plot_traits(acf, T_seen, cases, mode, hdr, dir_name, a2)
        plot_mean_by_case(acf, T_seen, cases, T_colors, mode, hdr, dir_name, a2)
        plot_mean_by_T(acf, acf0, T_seen, cases, mode, hdr, dir_name, a2)
    plot_trace(tr, tr0, T_seen, cases, hdr, dir_name, a2)

print("\nDone.")
