# -*- coding: utf-8 -*-
r"""
Plot the PLEIOTROPIC CROSS TERM of the multi-trait selection response.

PLOTTING ONLY -- this script runs no simulation.  It reads the cross_term_data_*.npz
files written by sweep_T_4cases_violin.py, which is the single simulation script for
the whole project.  (Its predecessor, cross_term_ndim.py, re-ran the dynamics itself;
that duplication is gone, so the figures here and the violin/hist/rank figures now
come from the very same runs.)

The board / paper Eq. (20) decomposition.  With the genetic (co)variance matrix

    G_{m l} = 2 sum_i a_{im} a_{il} p_i (1 - p_i)

the response of the mean phenotype of trait m is

    dzbar_m/dt = (1/V_s) [ G_mm delta_m  +  sum_{l != m} G_ml delta_l ]
                           \___________/    \____________________/
                              OWN term            CROSS term

which is the quantity the paper says makes the multi-trait model "not expressible in
terms of the V_g for each trait".

Outputs (one set per (direction, T, case) present in OUTDIR):
  fig1abc_cross_<tag>.pdf     -- Fig-1 A/B/C style panels, + D own vs cross over time,
                                 + E per-locus  a_i . delta  over time
  cross_over_trait_<tag>.pdf  -- vs TRAIT INDEX: (a) cross term, (b) per-locus
                                 a_{im} delta_m, (c) own vs cross magnitude
  cross_over_T_<dir>_a2_<a2>.pdf          -- vs NUMBER OF TRAITS, all cases
  cross_vs_own_summary_<dir>_a2_<a2>.pdf  -- cross/own importance vs T, all cases
"""

import os
import glob
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── output directory ──────────────────────────────────────────────────────────
# Must match sweep_T_4cases_violin.py: this is where its .npz files were written and
# where these figures are put.
RESULTS_DIR = 'results Aug 10'
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', RESULTS_DIR)
def out(name):  return os.path.join(OUTDIR, name)

# Per-(T, case) figures are made only for these T; every T found still enters the
# summary figures.  T=1 (cross term identically zero) and T=2 add nothing as panels.
FIG_T = [5, 20, 100]

CASE_LIST = ['A', 'B', 'C', 'D']
case_labels = {
    'A': r'A: $\Sigma_{ii}=\sigma^2,\ \Sigma_{ij}=+\sigma^2$',
    'B': r'B: $\Sigma_{ii}=\sigma^2,\ \Sigma_{ij}=-\sigma^2$',
    'C': r'C: $\Sigma_{ii}=\sigma^2/T,\ \Sigma_{ij}=+\sigma^2/T$',
    'D': r'D: $\Sigma_{ii}=\sigma^2/T,\ \Sigma_{ij}=-\sigma^2/T$',
}
case_colors = {'A': 'C0', 'B': 'C3', 'C': 'C2', 'D': 'C1', '0': 'k'}

# ZOOM_VIEWS lists the generation windows to render; each entry produces one figure in
# which ALL panels are restricted to that window, so they never disagree.  None = the
# whole run; a tuple = a close-up, which is what makes panel A legible, since 30000
# generations of L trajectories overplot into a smear.
ZOOM_VIEWS = [None, (25000, 30000)]


def load(path):
    """Read one cross_term_data_*.npz into a plain dict (arrays stay lazy-free)."""
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


# ── figure 1: Fig-1 A/B/C style panels, cross term, and a_i . delta ───────────
def fig1_style(d, tag, zoom=None):
    """`zoom` is None (whole run) or a (first_gen, last_gen) window applied to all
    panels at once."""
    BURN_IN, V_s, L = d['BURN_IN'], d['V_s'], d['L']
    m = d['gen'] >= BURN_IN
    m0 = d['gen0'] >= BURN_IN
    # sharex: all five panels cover the same generations (see the ZOOM note above).
    fig, axes = plt.subplots(5, 1, figsize=(10, 15), sharex=True)

    def masked(Pm):
        M = np.asarray(Pm, dtype=float).copy()
        M[(M <= 0) | (M >= 1)] = np.nan
        return M

    # A: allele frequencies (loci at p=0,1 not drawn), as in Fig 1A
    ax = axes[0]
    ax.plot(d['gen0'], masked(d['p0']), color='k',    lw=0.35, alpha=0.35)
    ax.plot(d['gen'],  masked(d['p']),  color='0.45', lw=0.45, alpha=0.55)
    ax.axvspan(0, BURN_IN, color='0.88', zorder=0)
    if zoom is None:
        ax.text(BURN_IN / 2, 0.97,
                f'burn-in (discarded from all\nstatistics), {BURN_IN} gens',
                transform=ax.get_xaxis_transform(), ha='center', va='top',
                fontsize=7.5, color='0.30')
    ax.set_ylim(0, 1)
    ax.set_ylabel('Frequency')
    ax.set_title(f'A  Allele frequencies at all L = {L} loci  '
                 r'(grey $\sigma^2>0$, black $\sigma^2=0$; loci at $p=0,1$ not drawn)',
                 fontsize=10, loc='left')

    # B: heritability of the focal trait over time, with histogram inset (Fig 1B)
    ax = axes[1]
    h2  = d['Vg'][:, 0]  / (1 + d['Vg'][:, 0])
    h20 = d['Vg0'][:, 0] / (1 + d['Vg0'][:, 0])
    ax.plot(d['gen'],  h2,  color='0.45', lw=0.7, label=r'$\sigma^2>0$')
    ax.plot(d['gen0'], h20, color='k',    lw=0.7, label=r'$\sigma^2=0$')
    ax.axvspan(0, BURN_IN, color='0.88', zorder=0)
    ax.set_ylabel(r'$h^2$ (trait 1)')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_title(r'B  Heritability of trait 1 over time (burn-in shaded).  '
                 'Inset: histogram over post-burn-in generations',
                 fontsize=10, loc='left')
    axi = ax.inset_axes([0.62, 0.12, 0.34, 0.42])
    axi.hist(h2[m],   bins=40, color='0.45', alpha=0.8)
    axi.hist(h20[m0], bins=40, color='k',    alpha=0.5)
    axi.tick_params(labelsize=6)

    # C: optimum displacement delta over time (Fig 1C)
    ax = axes[2]
    # blue / orange / black: black stays with sigma^2=0, matching panels A, B and E.
    ax.plot(d['gen'], d['delta'][:, 0], color='C0', lw=0.7, label=r'$\delta_1$')
    ax.plot(d['gen'], np.linalg.norm(d['delta'], axis=1), color='C1', lw=0.7,
            label=r'$\|\delta\|$')
    ax.plot(d['gen0'], d['delta0'][:, 0], color='k', lw=0.7,
            label=r'$\delta_1,\ \sigma^2=0$')
    ax.axhline(0, color='0.6', lw=0.5)
    ax.axvspan(0, BURN_IN, color='0.88', zorder=0)
    ax.set_ylabel(r'$\delta$')
    ax.legend(fontsize=8, loc='upper right', ncol=3)
    ax.set_title(r'C  Optimum displacement $\delta_t$ over time', fontsize=10, loc='left')

    # D: own vs cross contribution to dzbar_1/dt
    ax = axes[3]
    ax.plot(d['gen'], d['own'][:, 0] / V_s, color='C0', lw=0.7,
            label=r'own $G_{11}\delta_1/V_s$')
    ax.plot(d['gen'], d['cross'][:, 0] / V_s, color='C1', lw=0.7,
            label=r'cross $\sum_{l\neq1}G_{1l}\delta_l/V_s$')
    ax.axhline(0, color='0.6', lw=0.5)
    ax.axvspan(0, BURN_IN, color='0.88', zorder=0)
    ax.set_ylabel(r'contribution to $d\bar z_1/dt$')
    ax.legend(fontsize=8, loc='upper right')
    rms_own   = np.sqrt(np.mean(d['own'][m, 0] ** 2))
    rms_cross = np.sqrt(np.mean(d['cross'][m, 0] ** 2))
    ax.set_title('D  Own vs pleiotropic CROSS term for trait 1  '
                 f'(post burn-in RMS: own {rms_own:.3g}, cross {rms_cross:.3g}, '
                 f'ratio {rms_cross / rms_own:.2f})',
                 fontsize=10, loc='left')

    # E: per-locus a_i . delta over time.  Only segregating loci are drawn: the effect
    # row of a locus at p=0 or p=1 is stale (it is overwritten by the next mutation),
    # so its dot product is not part of the dynamics.
    ax = axes[4]
    def masked_w(w, p):
        W = np.asarray(w, dtype=float).copy()
        W[(np.asarray(p) <= 0) | (np.asarray(p) >= 1)] = np.nan
        return W
    Wf = masked_w(d['w'],  d['p'])
    W0 = masked_w(d['w0'], d['p0'])
    ax.plot(d['gen0'], W0, color='k',    lw=0.35, alpha=0.35)
    ax.plot(d['gen'],  Wf, color='0.45', lw=0.45, alpha=0.55)
    with np.errstate(invalid='ignore'):
        # all-NaN rows (no segregating locus) are legitimate early in the burn-in
        mean_w = np.where(np.isnan(Wf).all(axis=1), np.nan, np.nanmean(Wf, axis=1))
    ax.plot(d['gen'], mean_w, color='C0', lw=0.9, label='mean over segregating loci')
    ax.axhline(0, color='0.6', lw=0.5)
    ax.axvspan(0, BURN_IN, color='0.88', zorder=0)
    ax.set_xlabel('Generation')
    ax.set_ylabel(r'$\vec{a}_i\cdot\vec{\delta}$')
    ax.legend(fontsize=8, loc='upper right')
    rms_w = np.sqrt(np.nanmean(Wf[m] ** 2))
    ax.set_title(r'E  Per-locus $\vec{a}_i\cdot\vec{\delta}$ over time  '
                 r'(grey $\sigma^2>0$, black $\sigma^2=0$; segregating loci only; '
                 f'post burn-in RMS {rms_w:.3g})',
                 fontsize=10, loc='left')

    axes[0].set_xlim(*(zoom if zoom else (0, d['gen'].max())))

    fig.suptitle(f"Selection-response decomposition, T = {d['T']}, case {d['case']} "
                 f"({case_labels[d['case']]})\n{param_str(d)}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    suffix = '' if zoom is None else f'_gen{zoom[0]}-{zoom[1]}'
    fig.savefig(out(f'fig1abc_cross_{tag}{suffix}.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved fig1abc_cross_{tag}{suffix}.pdf")


# ── figure 2: quantities as a function of TRAIT INDEX within one run ──────────
def cross_over_trait_index(d, tag):
    T, V_s, case = d['T'], d['V_s'], d['case']
    m = d['gen'] >= d['BURN_IN']
    # Divide by V_s so the y-axes are contributions to dzbar_m/dt, the same units as
    # panel D of fig1abc_cross_*.pdf; without this the two figures differ by V_s.
    cross, own = d['cross'][m] / V_s, d['own'][m] / V_s
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    def violin(ax, vals):
        """violinplot, but tolerant of a degenerate (constant) series."""
        keep = [j for j, v in enumerate(vals) if np.ptp(v) > 0]
        if keep:
            parts = ax.violinplot([vals[j] for j in keep],
                                  positions=np.array(keep) + 1, widths=0.8,
                                  showmeans=True)
            for pc in parts['bodies']:
                pc.set_color(case_colors[case]); pc.set_alpha(0.55)
            for k in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
                parts[k].set_color(case_colors[case])
        for j, v in enumerate(vals):                      # constant series: a tick
            if np.ptp(v) == 0:
                ax.plot(j + 1, v[0] if len(v) else 0.0, marker='_', ms=10,
                        color=case_colors[case])

    # (a) distribution of the cross term per trait, over post-burn-in generations
    ax = axes[0]
    violin(ax, [cross[:, j] for j in range(T)])
    ax.axhline(0, color='0.6', lw=0.6)
    ax.set_xlabel('trait index $m$')
    ax.set_ylabel(r'cross term $\sum_{l\neq m}G_{ml}\delta_l\,/\,V_s$')
    ax.set_title('cross term per trait, pooled over\npost-burn-in generations',
                 fontsize=10)

    # (b) the per-locus, per-trait product a_{im} delta_m -- the quantity that enters
    # the selection kernel one trait at a time, and whose sum over m is a_i . delta
    # (panel E of the companion figure).  Pooled over segregating loci x recordings.
    ax = axes[1]
    seg = (d['tp'] > 0) & (d['tp'] < 1)                   # (n_tr, L)
    violin(ax, [np.asarray(d['ad'][:, :, j][seg], dtype=float) for j in range(T)])
    ax.axhline(0, color='0.6', lw=0.6)
    ax.set_xlabel('trait index $m$')
    ax.set_ylabel(r'$a_{im}\,\delta_m$')
    # trait index is an integer label, so half-integer ticks would be meaningless
    ax.xaxis.set_major_locator(plt.matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_title(r'per-locus $a_{im}\,\delta_m$ per trait, pooled over'
                 '\nsegregating loci and post-burn-in generations', fontsize=10)

    # (c) how big is the cross term relative to the own term, per trait
    ax = axes[2]
    rms_c = np.sqrt(np.mean(cross ** 2, axis=0))
    rms_o = np.sqrt(np.mean(own ** 2, axis=0))
    ax.bar(np.arange(1, T + 1) - 0.2, rms_o, width=0.4, color='C0', label='own RMS')
    ax.bar(np.arange(1, T + 1) + 0.2, rms_c, width=0.4, color='C1', label='cross RMS')
    ax.set_xlabel('trait index $m$')
    ax.set_ylabel(r'RMS contribution to $d\bar z_m/dt$')
    ax.set_yscale('log'); ax.legend(fontsize=8)
    ax.set_title(f'own vs cross magnitude per trait\n(mean ratio '
                 f'{np.mean(rms_c / rms_o):.2f})', fontsize=10)

    # The number of traits is fixed within a run, so it is not an axis here -- it is
    # annotated in each panel instead, so any single panel stays self-explanatory when
    # cropped out of the figure.
    for axx in axes:
        axx.text(0.02, 0.97, f'$T = {T}$ traits\ncase {case}', transform=axx.transAxes,
                 ha='left', va='top', fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='0.7', alpha=0.85))

    fig.suptitle(f'Pleiotropic cross term over trait index, T = {T}, case {case}  '
                 f'({case_labels[case]})\n{param_str(d)}', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(out(f'cross_over_trait_{tag}.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved cross_over_trait_{tag}.pdf")


# ── figure 3: the cross term as a function of the NUMBER OF TRAITS T ──────────
def cross_over_T(store, T_list, dir_name, a2, hdr):
    """store[(case, T)] = dict(cross=..., own=...) for trait 1, post burn-in."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.0))
    x_scale, width = 8.0, 0.45
    displ = np.linspace(-0.60, 0.60, len(CASE_LIST))

    # (a) distribution of the cross term itself, per T, all cases side by side
    ax = axes[0]
    for ci, case in enumerate(CASE_LIST):
        for T in T_list:
            vals = store[(case, T)]['cross']
            if np.ptp(vals) == 0:           # T=1: violinplot cannot take a constant
                ax.plot(np.log10(T) * x_scale + displ[ci], 0.0, marker='_',
                        ms=10, color=case_colors[case])
                continue
            parts = ax.violinplot(vals, positions=[np.log10(T) * x_scale + displ[ci]],
                                  widths=width, showmeans=True)
            for pc in parts['bodies']:
                pc.set_color(case_colors[case]); pc.set_alpha(0.55)
            for k in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
                parts[k].set_color(case_colors[case])
    ax.axhline(0, color='0.6', lw=0.6)
    ax.set_xticks([np.log10(T) * x_scale for T in T_list])
    ax.set_xticklabels([str(T) for T in T_list])
    ax.set_xlabel('Number of traits $T$')
    ax.set_ylabel(r'cross term $\sum_{l\neq1}G_{1l}\delta_l$')
    ax.set_title('cross term for trait 1 vs number of traits\n'
                 '(pooled over post-burn-in generations)', fontsize=10)

    # (b) RMS magnitude of both terms vs T
    ax = axes[1]
    # T=1 is dropped from the cross curve only: its cross term is zero by construction
    # (round-off, ~1e-17), which on a log axis would drag the panel down 14 decades.
    T_cross = [T for T in T_list if T > 1]
    for case in CASE_LIST:
        ro = [np.sqrt(np.mean(store[(case, T)]['own'] ** 2)) for T in T_list]
        rc = [np.sqrt(np.mean(store[(case, T)]['cross'] ** 2)) for T in T_cross]
        ax.plot(T_list, ro, marker='o', ls='--', color=case_colors[case], alpha=0.6,
                label=f'{case} own')
        ax.plot(T_cross, rc, marker='s', ls='-', color=case_colors[case],
                label=f'{case} cross')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xticks(T_list); ax.set_xticklabels([str(T) for T in T_list])
    ax.set_xlabel('Number of traits $T$'); ax.set_ylabel('RMS contribution')
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    ax.set_title('own (dashed) vs cross (solid) magnitude\n'
                 'vs number of traits  (cross omitted at T=1: it is zero)', fontsize=10)

    # (c) ratio vs T -- the headline number
    ax = axes[2]
    for case in CASE_LIST:
        ys = [np.sqrt(np.mean(store[(case, T)]['cross'] ** 2))
              / np.sqrt(np.mean(store[(case, T)]['own'] ** 2)) for T in T_list]
        ax.plot(T_list, ys, marker='o', color=case_colors[case], label=case_labels[case])
    ax.axhline(1, color='0.6', ls='--', lw=0.8)
    ax.set_xscale('log'); ax.set_xticks(T_list)
    ax.set_xticklabels([str(T) for T in T_list])
    ax.set_xlabel('Number of traits $T$'); ax.set_ylabel('RMS cross / RMS own')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    ax.set_title('pleiotropic share of the trait-1 response\n'
                 '(above 1 = cross term dominates)', fontsize=10)

    fig.suptitle('Pleiotropic cross term vs NUMBER OF TRAITS  '
                 f'({hdr}; trait 1, post burn-in)', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out(f'cross_over_T_{dir_name}_a2_{a2:.2f}.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved cross_over_T_{dir_name}_a2_{a2:.2f}.pdf")


def summary_fig(summary, T_list, dir_name, a2, hdr):
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for case in CASE_LIST:
        ax.plot(T_list, [summary[(case, T)] for T in T_list], marker='o',
                color=case_colors[case], label=case_labels[case])
    ax.axhline(1, color='0.6', ls='--', lw=0.8)
    ax.set_xscale('log'); ax.set_xticks(T_list)
    ax.set_xticklabels([str(t) for t in T_list])
    ax.set_xlabel('Number of traits $T$')
    ax.set_ylabel('RMS cross / RMS own  (trait 1)')
    ax.set_title('How much of the trait-1 selection response is pleiotropic?\n'
                 f'{hdr}, post burn-in', fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out(f'cross_vs_own_summary_{dir_name}_a2_{a2:.2f}.pdf'),
                bbox_inches='tight')
    plt.close(fig)
    print(f"Saved cross_vs_own_summary_{dir_name}_a2_{a2:.2f}.pdf")


# ── main: one pass over whatever the simulation produced ─────────────────────
PATTERN = re.compile(r'cross_term_data_(?P<dir>\w+?)_T(?P<T>\d+)_'
                     r'case(?P<case>[A-D])_a2_(?P<a2>[\d.]+)\.npz$')

files = sorted(glob.glob(out('cross_term_data_*.npz')))
if not files:
    raise SystemExit(f'No cross_term_data_*.npz in {OUTDIR}.\n'
                     'Run sweep_T_4cases_violin.py first -- it is the only script '
                     'that simulates.')

# group by (direction, a2) so each summary figure mixes only comparable runs
groups = {}
for f in files:
    mt = PATTERN.search(os.path.basename(f))
    if mt:
        groups.setdefault((mt['dir'], float(mt['a2'])), []).append((int(mt['T']),
                                                                    mt['case'], f))

for (dir_name, a2), entries in sorted(groups.items()):
    print(f"\n############## dir = {dir_name}, a2 = {a2:.2f} ##############")
    summary, store, hdr = {}, {}, ''
    T_seen = sorted({T for T, _, _ in entries})

    for T, case, f in sorted(entries):
        d = load(f)
        tag = f'{dir_name}_T{T}_case{case}_a2_{a2:.2f}'
        hdr = param_str(d)
        m = d['gen'] >= d['BURN_IN']
        rms_o = np.sqrt(np.mean(d['own'][m, 0] ** 2))
        rms_c = np.sqrt(np.mean(d['cross'][m, 0] ** 2))
        summary[(case, T)] = rms_c / rms_o
        store[(case, T)] = dict(own=d['own'][m, 0], cross=d['cross'][m, 0])
        print(f"  T={T:>3} case {case}: RMS own = {rms_o:.4g}, "
              f"RMS cross = {rms_c:.4g}, ratio = {rms_c / rms_o:.3f}")

        if T in FIG_T:
            for zoom in ZOOM_VIEWS:
                fig1_style(d, tag, zoom)
            cross_over_trait_index(d, tag)

    # summary figures need every case at every T; skip them if the sweep is partial
    missing = [(c, T) for T in T_seen for c in CASE_LIST if (c, T) not in store]
    if missing:
        print(f"  Skipping the summary figures: missing {len(missing)} (case, T) "
              f"combinations, e.g. {missing[:3]}")
        continue
    cross_over_T(store, T_seen, dir_name, a2, hdr)
    summary_fig(summary, T_seen, dir_name, a2, hdr)

print("\nDone.")
