# -*- coding: utf-8 -*-
"""
Vectorised sweep over T (number of trait dimensions) for the 4 covariance
cases.  THIS IS THE SINGLE SIMULATION SCRIPT: for each (T, case) it runs `rep`
replicate populations (vectorised over numpy arrays) and saves the per-locus
final state (a_{1,l}^2 and p_l).  It produces the h^2 violin plot, and the
saved per-locus data file is also consumed by the histogram scripts
(sweep_T_4cases_hist.py, hist_a1sq_pq.py, *_summary.py), so the expensive
30000-generation simulation is run only once.

Output (one set per a2 value):
  hist_T_4cases_data_<tag>.npz  -- per-locus a_{1,l}^2 and p_l (read by hist scripts)
  Vg_sweep_T_4cases_<tag>.npz   -- final Vg per (case, T, replicate) (derived)
  violin_T_4cases_<tag>.pdf     -- violin plot, h^2 vs T, 4 cases side-by-side
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools
import time

# ── parameters ────────────────────────────────────────────────────────────────
L         = 100
N         = 10000
V_s       = 5
mu        = 6.6e-6
theta     = 0.0
sigma_e2  = 1e-3
maxiter   = 30000
rep       = 100            # replicates per (T, case)
T_list    = [1, 5, 20, 100]
# a2 (mean effect size) swept; currently a single value
a2_values = np.array([0.03])
a2 = a2_values[0]          # current value (overwritten inside the sweep loop below)

# ── distributions for the per-mutation effect scale A ───────────────────────────
# Each draw_A(a2, T, n) returns n freshly drawn scales A (one per mutation event).
# Given A, each trait effect is a_t = sqrt(A/T) * N(0,1), so A = E[||a||^2 | A].
#   exp/const/gamma/lognormal all have mean a2 -> E[||a||^2]=a2 invariant in T.
#   twopoint is the user-specified PMF and is NOT mean-a2 / NOT T-invariant:
#   E[A] = (a2/T)(2 - 1/T) -> shrinks toward 0 as T grows.
def draw_const(a2, T, n):  return np.full(n, a2)                          # mean a2 (no variance)
# ── complex A-scale distributions (disabled) ───────────────────────────────────
# Simplified to a constant scale A = a2; the variable-scale distributions below are
# kept for reference but commented out. Re-enable a line in `dists` to use one.
# def draw_exp(a2, T, n):    return np.random.exponential(a2, size=n)      # mean a2
# def draw_gamma(a2, T, n):                                               # mean a2, shape k
#     k = 2.0
#     return np.random.gamma(k, a2 / k, size=n)
# def draw_lognormal(a2, T, n):                                           # mean a2
#     s = 1.0
#     return np.random.lognormal(np.log(a2) - 0.5 * s**2, s, size=n)
# def draw_twopoint(a2, T, n):                                            # a2 w.p. 1/T else a2/T
#     return np.where(np.random.rand(n) < 1.0 / T, a2, a2 / T)

dists = {
    'const':     draw_const,
    # 'twopoint':  draw_twopoint,
    # 'exp':       draw_exp,
    # 'gamma':     draw_gamma,
    # 'lognormal': draw_lognormal,
}

# ── per-trait effect scaling a_t ~ N(0, A / T**p) ───────────────────────────────
# p controls how each trait effect shrinks with the number of dimensions T.
#   aT1   (p=1)   : a_t ~ N(0, A/T)        -> E[||a||^2]=E[A]   (T-invariant)
#   aTsqrt(p=0.5) : a_t ~ N(0, A/sqrt(T))  -> E[||a||^2]=E[A]*sqrt(T) (grows with T)
a1_scalings = {
    'aT1':    1.0,
    'aTsqrt': 0.5,
}

# ── simulation core (vectorised over replicates, like the MPI version) ───────
def pmap(rho):  return rho / (1 + rho)
def rhomap(p):  return p / (1 - p)

def p_prime_sel_opt(p, delt, effects, V_s):
    """
    p:       (L, rep)
    delt:    (T, rep)
    effects: (L, rep, T)
    """
    S = 1 / (2 * V_s)
    dot_term = np.einsum('lrt,tr->lr', effects, delt)
    norm2    = np.sum(effects**2, axis=2)
    expo = 2 * S * (dot_term + 0.5 * norm2 * (2 * p - 1))
    return pmap(rhomap(p) * np.exp(expo))


def make_cov_matrix(sigma_e2, T, diag_scale, off_sign, off_scale):
    diag_val = sigma_e2 if diag_scale == 'full' else sigma_e2 / T
    off_mag  = sigma_e2 if off_scale  == 'full' else sigma_e2 / T
    cov = np.full((T, T), off_sign * off_mag, dtype=float)
    np.fill_diagonal(cov, diag_val)
    return cov


def chol_or_svd(cov):
    """Return L such that L L^T ~= cov, even when cov is rank-deficient."""
    cov_reg = (cov + cov.T) / 2 + 1e-10 * np.eye(cov.shape[0])
    try:
        return np.linalg.cholesky(cov_reg)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(cov_reg)
        eigvals = np.maximum(eigvals, 0)
        return eigvecs @ np.diag(np.sqrt(eigvals))


def simulate_vec(T, cov, rep, draw_A, texp):
    """Run `rep` replicates in parallel; return per-locus (a1_sq, p), each (L, rep).

    a1_sq = focal-trait squared effect a_{1,l}^2.  V_g(trait 1) is recovered as
    2 * sum_l a1_sq * p (1-p); returning the per-locus arrays (rather than just
    V_g) also lets the histogram scripts reuse this single simulation run.
    """
    # Effect scale A = a2 (constant; `draw_const`): the variable-scale distributions are
    # disabled, so A is the same for every mutation. Given A, each trait effect is
    # a_t = sqrt(A / T**texp) * N(0,1), drawn fresh per mutation and the SAME across the
    # T traits of a given mutation.  texp selects the per-trait scaling:
    #   texp=1   (aT1)    -> a_t ~ N(0, A/T),    E[||a||^2]=A,      invariant in T
    #   texp=0.5 (aTsqrt) -> a_t ~ N(0, A/sqrt(T)), E[||a||^2]=A*sqrt(T), grows with T
    # effects starts empty (population monomorphic, p=0); each mutation fills its own
    # (locus, rep) row.
    effects = np.zeros((L, rep, T))
    opt = np.zeros((T, rep))
    p   = np.zeros((L, rep))
    Lchol = chol_or_svd(cov)

    for t in range(maxiter):
        fixed_loci_1 = (p == 1)
        p[fixed_loci_1] = 0
        opt = opt - 2 * np.einsum('lr,lrt->tr', fixed_loci_1.astype(float), effects)

        allele_expected = 2 * p
        zbar = np.einsum('lr,lrt->tr', allele_expected, effects)

        # mutation: new alleles
        fixed_loci_0  = (p == 0)
        mutation_mask = (np.random.rand(L, rep) < N * mu) & fixed_loci_0
        np.place(p, mutation_mask, 1 / N)
        idx = np.where(mutation_mask)
        n_new = len(idx[0])
        # new mutation: scale A = a2 (constant), one per locus*rep event and SAME across
        # traits, plus a fresh N(0,1) direction.
        A_new = draw_A(a2, T, n_new)                            # (n_new,): constant scale a2 per mutation
        effects[idx[0], idx[1], :] = (np.random.normal(0, 1, size=(n_new, T))
                                      * np.sqrt(A_new / T**texp)[:, None])

        # mutation at polymorphic loci
        poly_loci = np.logical_not(fixed_loci_0) & (p < 1 - 1 / N)
        p[poly_loci] += (
            (np.random.rand(np.sum(poly_loci)) < N * mu * (1 - p[poly_loci])) / N
            - (np.random.rand(np.sum(poly_loci)) < N * mu * p[poly_loci]) / N
        )

        # selection + drift
        p = np.random.binomial(N, p_prime_sel_opt(p, opt - zbar, effects, V_s)) / N

        # optimum shift via cholesky factor (one sample per replicate)
        z = np.random.randn(T, rep)
        opt = (1 - theta) * opt + Lchol @ z

    # Return per-locus focal-trait data; V_g is recovered downstream as
    # 2 * sum_l a1_sq * p (1-p), which also feeds the histogram scripts.
    a1_sq = effects[:, :, 0] ** 2          # (L, rep): focal-trait squared effect
    return a1_sq, p


# ── main loop ────────────────────────────────────────────────────────────────
cases = {
    'A': dict(diag_scale='full',      off_sign=+1, off_scale='full'),
    'B': dict(diag_scale='full',      off_sign=-1, off_scale='full'),
    'C': dict(diag_scale='per_trait', off_sign=+1, off_scale='per_trait'),
    'D': dict(diag_scale='per_trait', off_sign=-1, off_scale='per_trait'),
}

# ── σ²=0 baseline (static optimum): denominator for the V_g ratio plot ─────────
# At σ²=0 the optimum never moves, so the covariance matrix is all zeros and the
# four cases A–D coincide. We therefore run just one simulation per (a1-scaling, T)
# using draw_const (A ≡ a2), and save the per-replicate focal-trait V_g. This is
# the *simulated* static-optimum baseline, to compare against the analytic
# Latter–Bulmer value 4·L·μ·V_s in plot_Vg_ratio_over_T.py.
print("\n############## σ²=0 baseline (static optimum, cases collapse) ##############")
for a2 in a2_values:                       # sets the global a2 read by simulate_vec
    base    = {'T_list': np.array(T_list)}   # totals    -> Vg_baseline_sigma0_*.npz
    base_pl = {'T_list': np.array(T_list)}   # per-locus -> hist_baseline_sigma0_*.npz
    for a1_name, texp in a1_scalings.items():
        Vg_T = np.zeros((len(T_list), rep))
        for ti, T in enumerate(T_list):
            t0 = time.time()
            cov0 = make_cov_matrix(0.0, T, diag_scale='full',
                                   off_sign=+1, off_scale='full')   # all zeros
            a1_sq, p = simulate_vec(T, cov0, rep, draw_const, texp)
            Vg_T[ti] = 2 * np.sum(a1_sq * p * (1 - p), axis=0)
            # keep per-locus arrays so the rank/hist scripts can overlay the baseline
            base_pl[f'{a1_name}_T{T}_a1sq'] = a1_sq
            base_pl[f'{a1_name}_T{T}_p']    = p
            print(f"  baseline a1={a1_name}  T={T}: mean Vg = {Vg_T[ti].mean():.5g}  "
                  f"[{time.time()-t0:.1f}s]")
        base[a1_name] = Vg_T
    np.savez(f'Vg_baseline_sigma0_a2_{a2:.2f}.npz', **base)
    np.savez(f'hist_baseline_sigma0_a2_{a2:.2f}.npz', **base_pl)
    print(f"Saved Vg_baseline_sigma0_a2_{a2:.2f}.npz and hist_baseline_sigma0_a2_{a2:.2f}.npz")

for (dist_name, draw_A), (a1_name, texp), a2 in itertools.product(
        dists.items(), a1_scalings.items(), a2_values):
    tag = f"{dist_name}_{a1_name}_a2_{a2:.2f}"
    print(f"\n############## dist = {dist_name}  a1 = {a1_name}  a2 = {a2:.3f}  ({tag}) ##############")

    # results[case][T_idx] = Vg array of length rep
    results = {label: np.zeros((len(T_list), rep)) for label in cases}
    # per-locus arrays saved for the histogram scripts (read by sweep_T_4cases_hist.py etc.)
    save_dict = {'T_list': np.array(T_list)}

    for ti, T in enumerate(T_list):
        print(f"\n=== T = {T} ===")
        for label, cfg in cases.items():
            t0 = time.time()
            cov = make_cov_matrix(sigma_e2, T, **cfg)
            a1_sq, p = simulate_vec(T, cov, rep, draw_A, texp)
            # focal-trait V_g per replicate = 2 * sum_l a_{1,l}^2 p_l(1-p_l)
            Vg = 2 * np.sum(a1_sq * p * (1 - p), axis=0)
            results[label][ti] = Vg
            # stash per-locus arrays so the histogram scripts can reuse this run
            save_dict[f'{label}_T{T}_a1sq'] = a1_sq
            save_dict[f'{label}_T{T}_p']    = p
            h2 = Vg / (1 + Vg)
            print(f"  Case {label}: mean h² = {h2.mean():.4f}  std = {h2.std():.4f}  "
                  f"[{time.time()-t0:.1f}s]")

    # ── save ──────────────────────────────────────────────────────────────────
    # (1) per-locus data — consumed by sweep_T_4cases_hist.py, hist_a1sq_pq.py, *_summary.py
    np.savez(f'hist_T_4cases_data_{tag}.npz', **save_dict)
    print(f"\nSaved hist_T_4cases_data_{tag}.npz")
    # (2) derived per-replicate Vg (convenience / downstream)
    np.savez(f'Vg_sweep_T_4cases_{tag}.npz',
             T_list=np.array(T_list),
             A=results['A'], B=results['B'], C=results['C'], D=results['D'])
    print(f"Saved Vg_sweep_T_4cases_{tag}.npz")

    # ── violin plot ──────────────────────────────────────────────────────────
    colors = {'A': 'C0', 'B': 'C3', 'C': 'C2', 'D': 'C1'}
    labels = {
        'A': r'A: $\Sigma_{ii}=\sigma^2,\ \Sigma_{ij}=+\sigma^2$',
        'B': r'B: $\Sigma_{ii}=\sigma^2,\ \Sigma_{ij}=-\sigma^2$',
        'C': r'C: $\Sigma_{ii}=\sigma^2/T,\ \Sigma_{ij}=+\sigma^2/T$',
        'D': r'D: $\Sigma_{ii}=\sigma^2/T,\ \Sigma_{ij}=-\sigma^2/T$',
    }

    fig, ax = plt.subplots(figsize=[14, 7])

    n_cases = 4
    # wider x-axis units (×8 instead of ×4) → more room per T group → fatter violins
    x_scale = 8.0
    displ   = np.linspace(-0.60, 0.60, n_cases)   # spread per case at each T
    width   = 0.45

    # compute y-range from the data so violins fill the panel
    all_h2 = np.concatenate([
        (results[c] / (1 + results[c])).ravel() for c in ['A', 'B', 'C', 'D']
    ])
    y_lo = max(0.0, all_h2.min() - 0.01)
    y_hi = all_h2.max() + 0.01

    for ci, label in enumerate(['A', 'B', 'C', 'D']):
        for ti, T in enumerate(T_list):
            Vg = results[label][ti]
            h2 = Vg / (1 + Vg)
            x  = np.log10(T) * x_scale + displ[ci]   # spread on a log-T axis
            parts = ax.violinplot(h2, positions=[x], widths=width, showmeans=True)
            for pc in parts['bodies']:
                pc.set_color(colors[label])
                pc.set_alpha(0.55)
            for k in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
                parts[k].set_color(colors[label])

    # legend
    handles = [plt.matplotlib.patches.Patch(color=colors[c], alpha=0.7,
                                            label=labels[c]) for c in cases]
    ax.legend(handles=handles, fontsize=11, loc='upper left')

    # ticks at T positions
    xticks = [np.log10(T) * x_scale for T in T_list]
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(T) for T in T_list], fontsize=11)
    ax.tick_params(axis='y', labelsize=11)

    ax.set_xlabel('Number of trait dimensions $T$', fontsize=13)
    ax.set_ylabel(r'Heritability $h^2$', fontsize=13)
    ax.set_title(r'Violin plot of $h^2$ across replicates ' +
                 f'(A~{dist_name}, $a_t$~{a1_name}, rep={rep}, $a^2={a2:.2f}$, $\\sigma^2=10^{{-3}}$, $V_s=5$, $N=10^4$, $L=100$)',
                 fontsize=12)
    ax.set_ylim([y_lo, y_hi])
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'violin_T_4cases_{tag}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"Saved violin_T_4cases_{tag}.pdf")
