# -*- coding: utf-8 -*-
"""
Per-locus diagnostic for the 4 covariance cases.

For each (T, case) pair, simulate `rep` replicates and record the final
per-locus quantities:
    a1_sq[locus, rep] = (effects[locus, rep, 0])**2    # first-dim squared effect
    p   [locus, rep] = allele frequency

For each T, produce one PDF with a 4 x 2 grid of histograms (rows = cases,
columns = [a_{1,l}^2, p_l]) pooled across all L loci and all replicates.

Output:
    hist_T_4cases_data.npz   -- a1_sq and p per (case, T, rep, locus)
    hist_T{T}_4cases.pdf     -- one PDF per T value
"""

import numpy as np
import matplotlib.pyplot as plt
import time

# ── parameters (match sweep_T_4cases_violin.py) ──────────────────────────────
L         = 100
N         = 10000
V_s       = 5
mu        = 6.6e-6
theta     = 0.0
sigma_e2  = 1e-3
maxiter   = 30000
rep       = 50            # replicates per (T, case); lower than violin script for speed
T_list    = [1, 2, 3, 5, 10, 20, 50, 100]
# Sweep over mutational variance a2 evenly from 0.01 to 0.1 (10 values).
a2_values = np.linspace(0.01, 0.1, 10)
a2 = a2_values[0]          # current value (overwritten inside the sweep loop below)

# ── simulation core (copied verbatim from sweep_T_4cases_violin.py) ──────────
def pmap(rho):  return rho / (1 + rho)
def rhomap(p):  return p / (1 - p)

def p_prime_sel_opt(p, delt, effects, V_s):
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
    cov_reg = (cov + cov.T) / 2 + 1e-10 * np.eye(cov.shape[0])
    try:
        return np.linalg.cholesky(cov_reg)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(cov_reg)
        eigvals = np.maximum(eigvals, 0)
        return eigvecs @ np.diag(np.sqrt(eigvals))


def simulate_vec(T, cov, rep):
    """Run `rep` replicates in parallel; return (a1_sq, p_final) per locus.

    a1_sq, p_final both have shape (L, rep).
    """
    # Effect scale A ~ Exponential(mean a2), drawn INDEPENDENTLY for each
    # (locus, replicate, trait).  Given A, a_t ~ N(0, A/T) (variance A/T).
    # => every (locus, trait) has its own size (a_t marginally Laplace);
    #    E[a_t^2]=a2/T, E[||a||^2]=a2.
    A = np.random.exponential(a2, size=(L, rep, T))
    effects = np.random.normal(0, 1, size=(L, rep, T)) * np.sqrt(A / T)
    opt = np.zeros((T, rep))
    p   = np.zeros((L, rep))
    Lchol = chol_or_svd(cov)

    for t in range(maxiter):
        fixed_loci_1 = (p == 1)
        p[fixed_loci_1] = 0
        opt = opt - 2 * np.einsum('lr,lrt->tr', fixed_loci_1.astype(float), effects)

        allele_expected = 2 * p
        zbar = np.einsum('lr,lrt->tr', allele_expected, effects)

        fixed_loci_0  = (p == 0)
        mutation_mask = (np.random.rand(L, rep) < N * mu) & fixed_loci_0
        np.place(p, mutation_mask, 1 / N)
        idx = np.where(mutation_mask)
        n_new = len(idx[0])
        # each new mutation: fresh per-trait scale A ~ Exp(a2), then N(0, A/T)
        A_new = np.random.exponential(a2, size=(n_new, T))
        effects[idx[0], idx[1], :] = (np.random.normal(0, 1, size=(n_new, T))
                                      * np.sqrt(A_new / T))

        poly_loci = np.logical_not(fixed_loci_0) & (p < 1 - 1 / N)
        p[poly_loci] += (
            (np.random.rand(np.sum(poly_loci)) < N * mu * (1 - p[poly_loci])) / N
            - (np.random.rand(np.sum(poly_loci)) < N * mu * p[poly_loci]) / N
        )

        p = np.random.binomial(N, p_prime_sel_opt(p, opt - zbar, effects, V_s)) / N

        z = np.random.randn(T, rep)
        opt = (1 - theta) * opt + Lchol @ z

    a1_sq = effects[:, :, 0] ** 2          # (L, rep): first-dim squared effect
    return a1_sq, p


# ── main loop ────────────────────────────────────────────────────────────────
cases = {
    'A': dict(diag_scale='full',      off_sign=+1, off_scale='full'),
    'B': dict(diag_scale='full',      off_sign=-1, off_scale='full'),
    'C': dict(diag_scale='per_trait', off_sign=+1, off_scale='per_trait'),
    'D': dict(diag_scale='per_trait', off_sign=-1, off_scale='per_trait'),
}

for a2 in a2_values:
    tag = f"a2_{a2:.2f}"
    print(f"\n############## a2 = {a2:.3f}  ({tag}) ##############")

    # data[case] -> dict keyed by T -> (a1_sq (L, rep), p (L, rep))
    data = {label: {} for label in cases}

    for ti, T in enumerate(T_list):
        print(f"\n=== T = {T} ===")
        for label, cfg in cases.items():
            t0 = time.time()
            cov = make_cov_matrix(sigma_e2, T, **cfg)
            a1_sq, p = simulate_vec(T, cov, rep)
            data[label][T] = (a1_sq, p)
            poly_frac = np.mean((p > 0) & (p < 1))
            print(f"  Case {label}: <a1^2>={a1_sq.mean():.3e}  "
                  f"<p>={p.mean():.3f}  polymorphic frac={poly_frac:.3f}  "
                  f"[{time.time()-t0:.1f}s]")

    # ── save raw data ────────────────────────────────────────────────────────
    save_dict = {'T_list': np.array(T_list)}
    for label in cases:
        for T in T_list:
            a1_sq, p = data[label][T]
            save_dict[f'{label}_T{T}_a1sq'] = a1_sq
            save_dict[f'{label}_T{T}_p']    = p
    np.savez(f'hist_T_4cases_data_{tag}.npz', **save_dict)
    print(f"\nSaved hist_T_4cases_data_{tag}.npz")

    # ── one figure per T ─────────────────────────────────────────────────────
    colors = {'A': 'C0', 'B': 'C3', 'C': 'C2', 'D': 'C1'}
    case_titles = {
        'A': r'A: $\Sigma_{ii}=\sigma^2,\ \Sigma_{ij}=+\sigma^2$',
        'B': r'B: $\Sigma_{ii}=\sigma^2,\ \Sigma_{ij}=-\sigma^2$',
        'C': r'C: $\Sigma_{ii}=\sigma^2/T,\ \Sigma_{ij}=+\sigma^2/T$',
        'D': r'D: $\Sigma_{ii}=\sigma^2/T,\ \Sigma_{ij}=-\sigma^2/T$',
    }

    for T in T_list:
        fig, axes = plt.subplots(4, 2, figsize=(10, 11), sharex='col')
        for ri, label in enumerate(['A', 'B', 'C', 'D']):
            a1_sq, p = data[label][T]
            a1_flat = a1_sq.ravel()
            p_flat  = p.ravel()

            ax_a = axes[ri, 0]
            ax_a.hist(a1_flat, bins=60, color=colors[label], alpha=0.75)
            ax_a.set_ylabel(case_titles[label], fontsize=9)
            ax_a.set_yscale('log')
            ax_a.grid(True, alpha=0.3)
            ax_a.axvline(a2 / T, color='k', ls='--', lw=0.8,
                         label=f'a²/T = {a2/T:.2e}')
            ax_a.legend(fontsize=7, loc='upper right')

            ax_p = axes[ri, 1]
            ax_p.hist(p_flat, bins=60, range=(0, 1), color=colors[label], alpha=0.75)
            ax_p.set_yscale('log')
            ax_p.grid(True, alpha=0.3)

        axes[0, 0].set_title(r'$a_{1,l}^2$  (first-dim squared effect)', fontsize=11)
        axes[0, 1].set_title(r'$p_l$  (allele frequency)', fontsize=11)
        axes[-1, 0].set_xlabel(r'$a_{1,l}^2$')
        axes[-1, 1].set_xlabel(r'$p_l$')

        fig.suptitle(f'Per-locus distributions, T = {T}, $a^2$ = {a2:.2f}  '
                     f'(L={L}, rep={rep}, pooled over all loci × reps)',
                     fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        fname = f'hist_T{T}_4cases_{tag}.pdf'
        plt.savefig(fname, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {fname}")
