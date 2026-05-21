# -*- coding: utf-8 -*-
"""
Vectorised sweep over T (number of trait dimensions) for the 4 covariance
cases.  For each (T, case) pair, simulate `rep` replicate populations in
parallel (no MPI -- vectorised over numpy arrays) and record the final
heritability of every replicate.  Saves data and produces a violin plot.

Output:
  Vg_sweep_T_4cases.npz   -- arrays of final Vg per (case, T, replicate)
  violin_T_4cases.pdf     -- violin plot, h^2 vs T, 4 cases side-by-side
"""

import numpy as np
import matplotlib.pyplot as plt
import time

# ── parameters ────────────────────────────────────────────────────────────────
L         = 100
N         = 10000
V_s       = 5
mu        = 6.6e-6
a2        = 0.1
theta     = 0.0
sigma_e2  = 1e-2
maxiter   = 30000
rep       = 50            # replicates per (T, case)
T_list    = [1, 2, 3, 5, 10, 20, 50, 100]

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


def simulate_vec(T, cov, rep):
    """Run `rep` replicates in parallel; return final Vg per replicate."""
    # Per-component std: each a_i ~ N(0, a2/T), so E[||a||^2] = a2 regardless of T.
    # Keeps total squared mutational size invariant in T (matches 1D when T=1).
    a_per_dim = np.sqrt(a2 / T)
    effects = np.random.normal(0, a_per_dim, size=(L, rep, T))
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
        effects[idx[0], idx[1], :] = np.random.normal(0, a_per_dim, size=(len(idx[0]), T))

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

    # No /T here: effects are already drawn with per-component variance a2/T,
    # so sum_t effects**2 ~ a2 per locus on average, and Vg is on the same scale as the 1D case.
    Vg = 2 * np.sum(np.sum(effects**2, axis=2) * p * (1 - p), axis=0)
    return Vg


# ── main loop ────────────────────────────────────────────────────────────────
cases = {
    'A': dict(diag_scale='full',      off_sign=+1, off_scale='full'),
    'B': dict(diag_scale='full',      off_sign=-1, off_scale='full'),
    'C': dict(diag_scale='per_trait', off_sign=+1, off_scale='per_trait'),
    'D': dict(diag_scale='per_trait', off_sign=-1, off_scale='per_trait'),
}

# results[case][T_idx] = Vg array of length rep
results = {label: np.zeros((len(T_list), rep)) for label in cases}

for ti, T in enumerate(T_list):
    print(f"\n=== T = {T} ===")
    for label, cfg in cases.items():
        t0 = time.time()
        cov = make_cov_matrix(sigma_e2, T, **cfg)
        Vg = simulate_vec(T, cov, rep)
        results[label][ti] = Vg
        h2 = Vg / (1 + Vg)
        print(f"  Case {label}: mean h² = {h2.mean():.4f}  std = {h2.std():.4f}  "
              f"[{time.time()-t0:.1f}s]")

# ── save ──────────────────────────────────────────────────────────────────────
np.savez('Vg_sweep_T_4cases.npz',
         T_list=np.array(T_list),
         A=results['A'], B=results['B'], C=results['C'], D=results['D'])
print("\nSaved Vg_sweep_T_4cases.npz")

# ── violin plot ───────────────────────────────────────────────────────────────
colors = {'A': 'C0', 'B': 'C3', 'C': 'C2', 'D': 'C1'}
labels = {
    'A': r'A: $\Sigma_{ii}=\sigma^2,\ \Sigma_{ij}=+\sigma^2$',
    'B': r'B: $\Sigma_{ii}=\sigma^2,\ \Sigma_{ij}=-\sigma^2$',
    'C': r'C: $\Sigma_{ii}=\sigma^2/T,\ \Sigma_{ij}=+\sigma^2/T$',
    'D': r'D: $\Sigma_{ii}=\sigma^2/T,\ \Sigma_{ij}=-\sigma^2/T$',
}

fig, ax = plt.subplots(figsize=[12, 5])

n_cases = 4
displ   = np.linspace(-0.30, 0.30, n_cases)   # spread per case at each T
width   = 0.18

for ci, label in enumerate(['A', 'B', 'C', 'D']):
    for ti, T in enumerate(T_list):
        Vg = results[label][ti]
        h2 = Vg / (1 + Vg)
        x  = np.log10(T) * 4 + displ[ci]   # spread on a log-T axis
        parts = ax.violinplot(h2, positions=[x], widths=width, showmeans=True)
        for pc in parts['bodies']:
            pc.set_color(colors[label])
            pc.set_alpha(0.55)
        for k in ('cbars', 'cmins', 'cmaxes', 'cmeans'):
            parts[k].set_color(colors[label])

# legend
handles = [plt.matplotlib.patches.Patch(color=colors[c], alpha=0.7,
                                        label=labels[c]) for c in cases]
ax.legend(handles=handles, fontsize=9, loc='upper left')

# ticks at T positions
xticks = [np.log10(T) * 4 for T in T_list]
ax.set_xticks(xticks)
ax.set_xticklabels([str(T) for T in T_list])

ax.set_xlabel('Number of trait dimensions $T$', fontsize=12)
ax.set_ylabel(r'Heritability $h^2$', fontsize=12)
ax.set_title(r'Violin plot of $h^2$ across replicates ' +
             f'(rep={rep}, $\\sigma^2=10^{{-2}}$, $V_s=5$, $N=10^4$, $L=100$)',
             fontsize=11)
ax.set_ylim([0, 1])
ax.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('violin_T_4cases.pdf', bbox_inches='tight')
print("Saved violin_T_4cases.pdf")
