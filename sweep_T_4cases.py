# -*- coding: utf-8 -*-
"""
Sweep over T (number of trait dimensions) for the 4 covariance-matrix cases.

For each (T, case) pair, run a single-replicate trajectory simulation and
record the mean heritability h^2 over the steady-state window.  Saves a
table to `h2_vs_T_4cases.txt` and produces `h2_vs_T_4cases.pdf`.

Cases:
  A: diag = sigma^2,    off = +sigma^2
  B: diag = sigma^2,    off = -sigma^2
  C: diag = sigma^2/T,  off = +sigma^2/T
  D: diag = sigma^2/T,  off = -sigma^2/T
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
maxiter   = 30000               # 3*N — enough to reach steady state
burn_in   = 15000               # discard first half
T_list    = [1, 2, 3, 5, 10, 20, 50, 100]

# ── helpers ──────────────────────────────────────────────────────────────────
def pmap(rho):  return rho / (1 + rho)
def rhomap(p):  return p / (1 - p)

def p_prime_sel_opt(p, delt_opt, effects, V_s):
    S = 1 / (2 * V_s)
    dot_term = np.einsum('lt,t->l', effects, delt_opt)
    norm2    = np.sum(effects**2, axis=1)
    expo = 2 * S * (dot_term + 0.5 * norm2 * (2 * p - 1))
    return pmap(rhomap(p) * np.exp(expo))


def make_cov_matrix(sigma_e2, T, diag_scale, off_sign, off_scale):
    diag_val = sigma_e2 if diag_scale == 'full' else sigma_e2 / T
    off_mag  = sigma_e2 if off_scale  == 'full' else sigma_e2 / T
    off_val  = off_sign * off_mag
    cov = np.full((T, T), off_val, dtype=float)
    np.fill_diagonal(cov, diag_val)
    return cov


def chol_or_svd(cov):
    """Return L such that L @ L.T ~= cov, even for rank-deficient/near-PSD."""
    cov_reg = (cov + cov.T) / 2 + 1e-12 * np.eye(cov.shape[0])
    try:
        return np.linalg.cholesky(cov_reg)
    except np.linalg.LinAlgError:
        U, S, _ = np.linalg.svd(cov_reg)
        S = np.maximum(S, 0)
        return U @ np.diag(np.sqrt(S))


def simulate_meanh2(T, cov):
    a = np.sqrt(a2)
    effects = np.random.normal(0, a, size=(L, T))
    opt = np.zeros(T)
    p   = np.zeros(L)

    Lchol = chol_or_svd(cov)
    Vg_hist = np.zeros(maxiter)

    for t in range(maxiter):
        fixed_loci_1 = (p == 1)
        p[fixed_loci_1] = 0
        opt = opt - 2 * np.einsum('l,lt->t', fixed_loci_1.astype(float), effects)

        allele_expected = 2 * p
        zbar = np.einsum('l,lt->t', allele_expected, effects)
        delt = opt - zbar

        Vg_hist[t] = 2 * np.sum(np.sum(effects**2, axis=1) * p * (1 - p)) / T

        # mutation: new alleles
        fixed_loci_0  = (p == 0)
        mutation_mask = (np.random.rand(L) < N * mu) & fixed_loci_0
        p[mutation_mask] = 1 / N
        new_idx = np.where(mutation_mask)
        effects[new_idx[0], :] = np.random.normal(0, a, size=(len(new_idx[0]), T))

        # mutation at polymorphic loci
        poly_loci = np.logical_not(fixed_loci_0) & (p < 1 - 1 / N)
        p[poly_loci] += (
            (np.random.rand(np.sum(poly_loci)) < N * mu * (1 - p[poly_loci])) / N
            - (np.random.rand(np.sum(poly_loci)) < N * mu * p[poly_loci]) / N
        )

        # selection + drift
        p = np.random.binomial(N, p_prime_sel_opt(p, delt, effects, V_s)) / N

        # optimum shift via cholesky
        noise = Lchol @ np.random.randn(T)
        opt = (1 - theta) * opt + noise

    Vg_steady = Vg_hist[burn_in:]
    h2_steady = Vg_steady / (1 + Vg_steady)
    return h2_steady.mean(), h2_steady.std()


# ── main loop ─────────────────────────────────────────────────────────────────
cases = {
    'A': dict(diag_scale='full',      off_sign=+1, off_scale='full'),
    'B': dict(diag_scale='full',      off_sign=-1, off_scale='full'),
    'C': dict(diag_scale='per_trait', off_sign=+1, off_scale='per_trait'),
    'D': dict(diag_scale='per_trait', off_sign=-1, off_scale='per_trait'),
}

results = {label: [] for label in cases}    # mean h² per T
results_std = {label: [] for label in cases}

for T in T_list:
    print(f"\n=== T = {T} ===")
    for label, cfg in cases.items():
        t0 = time.time()
        cov = make_cov_matrix(sigma_e2, T, **cfg)
        mean_h2, std_h2 = simulate_meanh2(T, cov)
        elapsed = time.time() - t0
        results[label].append(mean_h2)
        results_std[label].append(std_h2)
        print(f"  Case {label}: mean h² = {mean_h2:.4f}  (±{std_h2:.4f}) "
              f"[{elapsed:.1f}s]")

# ── save table ────────────────────────────────────────────────────────────────
header = "T  " + "  ".join([f"h2_{c}  std_{c}" for c in cases])
table = np.column_stack([T_list] +
                         [results[c] for c in cases] +
                         [results_std[c] for c in cases])
np.savetxt("h2_vs_T_4cases.txt", table, header=header, fmt="%.6g")
print("\nSaved h2_vs_T_4cases.txt")

# ── plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=[7, 5])
markers = {'A': 'o', 'B': 's', 'C': '^', 'D': 'D'}
colors  = {'A': 'C0', 'B': 'C3', 'C': 'C2', 'D': 'C1'}
labels  = {
    'A': r'A: $\Sigma_{ii}=\sigma^2,\ \Sigma_{ij}=+\sigma^2$',
    'B': r'B: $\Sigma_{ii}=\sigma^2,\ \Sigma_{ij}=-\sigma^2$',
    'C': r'C: $\Sigma_{ii}=\sigma^2/T,\ \Sigma_{ij}=+\sigma^2/T$',
    'D': r'D: $\Sigma_{ii}=\sigma^2/T,\ \Sigma_{ij}=-\sigma^2/T$',
}

for label in cases:
    ax.errorbar(T_list, results[label], yerr=results_std[label],
                marker=markers[label], color=colors[label],
                label=labels[label], capsize=3, markersize=7)

ax.set_xscale('log')
ax.set_xlabel('Number of trait dimensions $T$', fontsize=12)
ax.set_ylabel(r'Mean heritability $h^2$ (steady state)', fontsize=12)
ax.set_title(r'$h^2$ vs $T$ for 4 covariance-matrix cases  '
             r'($\sigma^2=10^{-2},\ V_s=5,\ N=10^4,\ L=100$)', fontsize=11)
ax.legend(fontsize=9, loc='best')
ax.grid(True, which='both', alpha=0.3)
ax.set_xticks(T_list)
ax.set_xticklabels([str(t) for t in T_list])

plt.tight_layout()
plt.savefig('h2_vs_T_4cases.pdf', bbox_inches='tight')
print("Saved h2_vs_T_4cases.pdf")
