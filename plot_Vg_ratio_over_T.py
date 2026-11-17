# -*- coding: utf-8 -*-
"""
Plot V_g(T) of trait 1, relative to the simulated static-optimum (sigma^2 = 0) baseline,
against the number of trait dimensions T for the covariance cases A-F, and compare it
with the earlier run.  PLOTTING ONLY: reads the files of sweep_T_4cases_violin.py.

NOV 16 (trait 1 neutral).  When the sweep is run with NEUTRAL_TRAITS = [0], a_1 = 0 in
the simulation and V_g(trait 1) is computed with a fixed squared effect,
V_g1 = a_1^2 * 2 sum_l p_l q_l (see the sweep script).  The analytic 4 L mu V_s / T
baseline does not apply to a neutral trait, so everything here is normalised by the
SIMULATED sigma^2 = 0 baseline of the same run -- the fixed a_1^2 cancels in the ratio:

    V_g(T) / V_g(T)|_{sigma^2 = 0}

COMPARISON WITH THE EARLIER RUN.  REF_DIR points at the earlier run with the same
parameters in which EVERY trait is under selection.  It is only read, never written.
  Vg_ratio_over_T_<tag>.pdf   V_g / simulated V_g|sigma^2=0 vs T
                              (solid = this run, dashed = earlier run)
  Vg_vs_before_<tag>.pdf      (a) raw V_g vs T, both runs, with both static baselines
                                  and the analytic 4 L mu V_s / T line
                              (b) V_g(this run) / V_g(earlier run) at the same T
"""

import os
import itertools
import numpy as np
import matplotlib.pyplot as plt

# ── output directory ──────────────────────────────────────────────────────────
# Read the .npz files written by sweep_T_4cases_violin.py, and put the figures next
# to them.  Must match the RESULTS_DIR set in that script.
RESULTS_DIR = 'results_Nov_16'
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', RESULTS_DIR)
def opath(name):  return os.path.join(OUTDIR, name)

# tags to look for (must match the sweep's dists / A1_TAG / a2_values)
a2_values   = np.array([0.03])
dist_names  = ['const']                 # complex A-scale dists disabled
# per-trait DIRECTION distributions (must match violin's `dir_dists` keys):
#   'gauss' -- a_t ~ N(0, A/T)   ; 'pm' -- a_t = +-sqrt(A/T), the paper's model
dir_names   = ['gauss', 'pm']
# per-trait scaling is fixed at a_t ~ sqrt(A/T) * direction; A1_TAG must match
# violin's `A1_TAG` since it appears in the filenames and baseline .npz keys.
A1_TAG = 'aT1'

cases   = ['A', 'B', 'C', 'D', 'E', 'F']
colors  = {'A': 'C0', 'B': 'C3', 'C': 'C2', 'D': 'C1', 'E': 'C4', 'F': 'C5'}
labels  = {
    'A':  r'A: $\Sigma_{ii}=\sigma^2,\ \Sigma_{ij}=+\sigma^2$',
    'B':  r'B: $\Sigma_{ii}=\sigma^2,\ \Sigma_{ij}=-\sigma^2$',
    'C':  r'C: $\Sigma_{ii}=\sigma^2/T,\ \Sigma_{ij}=+\sigma^2/T$',
    'D':  r'D: $\Sigma_{ii}=\sigma^2/T,\ \Sigma_{ij}=-\sigma^2/T$',
    'E':  r'E: $\Sigma_{ii}=\sigma^2/T,\ \Sigma_{ij}=0$',
    'F':  r'F: $\Sigma_{ii}=\sigma^2,\ \Sigma_{ij}=0$',
}

# ── reference run: same parameters, every trait under selection (read only) ───
REF_DIR   = 'results sep 15 OLD'        # L = 100; set to None to switch the overlay off
REF_LABEL = 'all traits under selection (Sep 15 run)'
def rpath(name):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', REF_DIR, name)

ALL = list(cases) + ['0']               # '0' = the sigma^2 = 0 baseline


def load_run(data_file, base_file):
    """One sweep: per-T mean V_g and SEM for every case and for the sigma^2=0 baseline
    (key '0').  Returns None if a file is missing."""
    if not (os.path.exists(data_file) and os.path.exists(base_file)):
        return None
    z = np.load(data_file)
    neutral = ([int(t) + 1 for t in z['neutral_traits']]
               if 'neutral_traits' in z.files else [])
    r = dict(z=z, T=np.asarray(z['T_list'], dtype=float), neutral=neutral,
             Vg1=4 * int(z['L']) * float(z['mu']) * float(z['V_s']))
    arr = {c: z[c] for c in cases if c in z.files}
    arr['0'] = np.load(base_file)[A1_TAG]
    r['mean'] = {c: v.mean(axis=1) for c, v in arr.items()}
    r['sem']  = {c: v.std(axis=1) / np.sqrt(v.shape[1]) for c, v in arr.items()}
    return r


def describe(r):
    if r['neutral']:
        return (rf"$a_t=0$ on trait(s) {r['neutral']} in the simulation; "
                r"$V_{g1}=a_1^2\cdot 2\sum_l p_l q_l$ with fixed $a_1^2$")
    return 'every trait under selection'


def params(r):
    z = r['z']
    return (rf"L={int(z['L'])}, N={int(z['N'])}, $\mu$={float(z['mu']):g}, "
            rf"$V_s$={float(z['V_s']):g}, $a^2$={float(z['a2']):g}, "
            rf"A~{z['dist_name']}, dir={z['dir_name']}, "
            rf"$\sigma^2$={float(z['sigma_e2']):g}, $\theta$={float(z['theta']):g}, "
            f"generations={int(z['maxiter'])}, burn-in={int(z['BURN_IN'])}, "
            f"{int(z['rep'])} reps x {int(z['n_snap'])} snapshots "
            f"(every {int(z['SAMPLE_EVERY'])} gens), error bars = SEM")


def log_T_axis(ax, ticks):
    ax.set_xscale('log')
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(int(t)) for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel('number of trait dimensions  T')
    ax.grid(alpha=0.3)


def style(c):
    return (dict(color='k', lw=1.6, marker='s') if c == '0'
            else dict(color=colors[c], marker='o'))


def ratio(a, sa, b, sb):
    """a / b with the SEM of the ratio (independent a, b)."""
    q = a / b
    return q, q * np.sqrt((sa / a) ** 2 + (sb / b) ** 2)


def table(title, T, rows):
    print(f"\n  {title}")
    print(f"  {'T':>5} " + ' '.join(f"{('sig0' if c == '0' else c):>8}" for c in rows))
    for ti, t in enumerate(T):
        print(f"  {int(t):>5} " + ' '.join(f'{rows[c][ti]:>8.3f}' for c in rows))


# ── figures, one pair per (dist, direction, a2) ───────────────────────────────
for dist_name, dir_name, a2 in itertools.product(
        dist_names, dir_names, a2_values):
    tag  = f"{dist_name}_{dir_name}_{A1_TAG}_a2_{a2:.2f}"
    data = f'Vg_sweep_T_4cases_{tag}.npz'
    base = f'Vg_baseline_sigma0_{dir_name}_a2_{a2:.2f}.npz'

    new = load_run(opath(data), opath(base))
    if new is None:
        print(f"[skip] {data} / {base} not found in {OUTDIR}. "
              "Run sweep_T_4cases_violin.py first.")
        continue
    ref = load_run(rpath(data), rpath(base)) if REF_DIR else None
    if REF_DIR and ref is None:
        print(f"[note] no reference run in {REF_DIR}; comparison omitted.")
    runs = [(new, '-', 1.0)] + ([(ref, '--', 0.5)] if ref else [])
    T_ticks = sorted(set(new['T']) | (set(ref['T']) if ref else set()))

    # ── figure 1: V_g / simulated V_g|sigma^2=0 vs T ──────────────────────────
    fig, ax = plt.subplots(figsize=[9.5, 7.5])
    for r, ls, alpha in runs:
        rows = {}
        for c in cases:
            if c not in r['mean']:
                continue
            q, e = ratio(r['mean'][c], r['sem'][c], r['mean']['0'], r['sem']['0'])
            rows[c] = q
            ax.errorbar(r['T'], q, yerr=e, capsize=3, ls=ls, alpha=alpha,
                        label=labels[c] if r is new else None, **style(c))
        table(f"{tag}  V_g / V_g|sigma^2=0   [{describe(r)}]", r['T'], rows)
    if ref:
        ax.plot([], [], color='0.4', ls='--', marker='o', alpha=0.6,
                label=f'dashed: {REF_LABEL}')
    ax.axhline(1.0, color='k', ls=':', lw=1, label=r'static optimum (=1)')
    log_T_axis(ax, T_ticks)
    ax.set_ylabel(r'$V_g(T)\,/\,V_g(T)|_{\sigma^2=0}$   (trait 1, simulated baseline)')
    ax.set_title('Genetic variance of trait 1 relative to the simulated static baseline\n'
                 f'solid: {describe(new)}\n'
                 + (f'dashed: {describe(ref)}\n' if ref else '')
                 + params(new), fontsize=8.5, wrap=True)
    ax.legend(fontsize=8, loc='best')
    fig.tight_layout()
    out = f'Vg_ratio_over_T_{tag}.pdf'
    fig.savefig(opath(out))
    print(f"Saved {out}")

    if not ref:
        continue

    # ── figure 2: this run against the earlier one ────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=[15, 6.4])

    # (a) raw V_g, with both simulated static baselines and the analytic line
    ax = axes[0]
    for r, ls, alpha in runs:
        for c in ALL:
            if c in r['mean']:
                lab = (r'$\sigma^2=0$ (simulated)' if c == '0' else labels[c])
                ax.errorbar(r['T'], r['mean'][c], yerr=r['sem'][c], capsize=3, ls=ls,
                            alpha=alpha, label=lab if r is new else None, **style(c))
    Tt = np.array(T_ticks)
    ax.plot(Tt, new['Vg1'] / Tt, color='0.5', ls='-.', lw=1.2,
            label=r'analytic $4L\mu V_s/T$ (all traits selected)')
    ax.plot([], [], color='0.4', ls='--', marker='o', alpha=0.6,
            label=f'dashed: {REF_LABEL}')
    ax.set_yscale('log')
    log_T_axis(ax, T_ticks)
    ax.set_ylabel(r'$V_g$ of trait 1')
    ax.set_title('(a) raw $V_g$ of trait 1', fontsize=10)
    ax.legend(fontsize=7)

    # (b) V_g(this run) / V_g(earlier run) at the T values both runs have
    ax = axes[1]
    common = sorted(set(new['T']) & set(ref['T']))
    i_new = [list(new['T']).index(t) for t in common]
    i_ref = [list(ref['T']).index(t) for t in common]
    rows = {}
    for c in ALL:
        if c in new['mean'] and c in ref['mean']:
            q, e = ratio(new['mean'][c][i_new], new['sem'][c][i_new],
                         ref['mean'][c][i_ref], ref['sem'][c][i_ref])
            rows[c] = q
            ax.errorbar(common, q, yerr=e, capsize=3,
                        label=(r'$\sigma^2=0$' if c == '0' else f'case {c}'), **style(c))
    table(f"{tag}  V_g(this run) / V_g(earlier run)", common, rows)
    ax.axhline(1.0, color='k', ls=':', lw=1)

    # house-of-cards prediction for the sigma^2=0 ratio: one trait fewer under
    # selection turns 4 L mu V_s / T into 4 L mu V_s / (T - n_neutral)
    Tc = np.array(common, dtype=float)
    n_neu = len(new['neutral'])
    hoc = Tc / (Tc - n_neu) if n_neu else np.ones_like(Tc)
    ax.plot(Tc, hoc, color='0.5', ls='--', lw=1.2, marker='x', ms=7,
            label=rf'$T/(T-{n_neu})$ (house-of-cards, $\sigma^2=0$)')

    # write the sigma^2=0 value (black) and the prediction (grey) next to each point
    if '0' in rows:
        for t, v, h in zip(Tc, rows['0'], hoc):
            ax.annotate(f'{v:.2f}', (t, v), xytext=(7, 4), textcoords='offset points',
                        fontsize=9, fontweight='bold', color='k')
            ax.annotate(f'{h:.2f}', (t, h), xytext=(7, -11), textcoords='offset points',
                        fontsize=8, color='0.4')
    ax.set_yscale('log')
    log_T_axis(ax, common)
    ax.set_ylabel(r'$V_g$(this run) / $V_g$(earlier run)')
    ax.set_title('(b) ratio at the same $T$  (numbers: black = $\\sigma^2=0$, '
                 'grey = $T/(T-1)$; full table below)', fontsize=10)
    ax.legend(fontsize=8)

    # every value of panel (b), as a table under it
    row_keys = ['0'] + [c for c in cases if c in rows]
    cell = [[f'{rows[c][j]:.2f}' for j in range(len(common))] for c in row_keys if c in rows]
    cell.append([f'{h:.2f}' for h in hoc])
    row_lab = [(r'$\sigma^2=0$' if c == '0' else f'case {c}') for c in row_keys if c in rows]
    row_lab.append(f'T/(T-{n_neu})')
    tb = ax.table(cellText=cell, rowLabels=row_lab,
                  colLabels=[f'T = {int(t)}' for t in common],
                  cellLoc='center', bbox=[0.0, -0.78, 1.0, 0.58])
    tb.auto_set_font_size(False)
    tb.set_fontsize(8)
    for (r_, c_), cl in tb.get_celld().items():
        if r_ >= 1 and r_ - 1 < len(row_keys) and row_keys[r_ - 1] in colors:
            cl.get_text().set_color(colors[row_keys[r_ - 1]])

    fig.suptitle(f'THIS RUN (solid): {describe(new)}\n'
                 f'EARLIER RUN (dashed): {describe(ref)}\n' + params(new), fontsize=9)
    fig.set_size_inches(15, 10)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.86, bottom=0.36, wspace=0.18)
    out = f'Vg_vs_before_{tag}.pdf'
    fig.savefig(opath(out))
    print(f"Saved {out}")

plt.show()
