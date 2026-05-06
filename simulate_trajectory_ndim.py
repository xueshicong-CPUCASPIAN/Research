# -*- coding: utf-8 -*-
"""
n-dimensional trajectory simulation with FULL COVARIANCE MATRIX for the
optimum noise.  Supports 4 cases of Sigma = Cov(Delta_opt):

  Case A:  diag = sigma^2,    off-diag = +sigma^2
  Case B:  diag = sigma^2,    off-diag = -sigma^2
  Case C:  diag = sigma^2/T,  off-diag = +sigma^2/T
  Case D:  diag = sigma^2/T,  off-diag = -sigma^2/T

Saves trajectory files for each case:
  hist_<case>_nd.txt        -- allele frequency trajectories  (maxiter, L)
  Vg_hist_<case>_nd.txt     -- genetic variance per generation
  delta_norm_<case>_nd.txt  -- ||opt - zbar||_2 per generation
  delta_1_<case>_nd.txt     -- first trait component of delta (signed)
"""

import numpy as np

# ── shared functions ─────────────────────────────────────────────────────────
def pmap(rho):
    return rho / (1 + rho)

def rhomap(p):
    return p / (1 - p)

def p_prime_sel_opt(p, delt_opt, effects, V_s):
    S = 1 / (2 * V_s)
    dot_term = np.einsum('lt,t->l', effects, delt_opt)
    norm2    = np.sum(effects**2, axis=1)
    expo = 2 * S * (dot_term + 0.5 * norm2 * (2 * p - 1))
    return pmap(rhomap(p) * np.exp(expo))


def make_cov_matrix(sigma_e2, n_traits, diag_scale, off_sign, off_scale):
    """
    Build T x T covariance matrix for the optimum noise.

      diag_scale : 'full' -> sigma_e2     ; 'per_trait' -> sigma_e2 / T
      off_scale  : 'full' -> sigma_e2     ; 'per_trait' -> sigma_e2 / T
      off_sign   : +1 or -1
    """
    diag_val = sigma_e2 if diag_scale == 'full' else sigma_e2 / n_traits
    off_mag  = sigma_e2 if off_scale  == 'full' else sigma_e2 / n_traits
    off_val  = off_sign * off_mag

    cov = np.full((n_traits, n_traits), off_val, dtype=float)
    np.fill_diagonal(cov, diag_val)
    return cov


# ── trajectory simulation (single replicate, n-D, with cov matrix) ───────────
def simulate_trajectory_nd(L, sigma_e2, N, V_s, mu, a2, theta, n_traits, cov_matrix):
    a = np.sqrt(a2)
    effects = np.random.normal(0, a, size=(L, n_traits))
    opt = np.zeros(n_traits)
    p   = np.zeros(L)
    maxiter = int(10 * N)

    hist       = np.zeros((maxiter, L))
    Vg_hist    = np.zeros(maxiter)
    delta_norm = np.zeros(maxiter)
    delta_1    = np.zeros(maxiter)

    mean_zero = np.zeros(n_traits)

    for t in range(maxiter):
        if t % 2000 == 0:
            print(f"    t = {t} / {maxiter}")

        fixed_loci_1 = (p == 1)
        p[fixed_loci_1] = 0
        opt = opt - 2 * np.einsum('l,lt->t', fixed_loci_1.astype(float), effects)

        allele_expected = 2 * p**2 + 2 * p * (1 - p)
        zbar = np.einsum('l,lt->t', allele_expected, effects)
        delt = opt - zbar

        hist[t, :]    = p
        Vg_hist[t]    = 2 * np.sum(np.sum(effects**2, axis=1) * p * (1 - p)) / n_traits
        delta_norm[t] = np.linalg.norm(delt)
        delta_1[t]    = delt[0]

        # mutation: new alleles enter at freq 1/N
        fixed_loci_0  = (p == 0)
        mutation_mask = (np.random.rand(L) < N * mu) & fixed_loci_0
        p[mutation_mask] = 1 / N
        new_idx = np.where(mutation_mask)
        effects[new_idx[0], :] = np.random.normal(0, a, size=(len(new_idx[0]), n_traits))

        # mutation at polymorphic loci
        poly_loci = np.logical_not(fixed_loci_0) & (p < 1 - 1 / N)
        p[poly_loci] += (
            (np.random.rand(np.sum(poly_loci)) < N * mu * (1 - p[poly_loci])) / N
            - (np.random.rand(np.sum(poly_loci)) < N * mu * p[poly_loci]) / N
        )

        # selection + drift
        p = np.random.binomial(N, p_prime_sel_opt(p, delt, effects, V_s)) / N

        # optimum shift using full covariance matrix
        if sigma_e2 > 0:
            noise = np.random.multivariate_normal(mean_zero, cov_matrix,
                                                   check_valid='ignore')
            opt = (1 - theta) * opt + noise

    return hist, Vg_hist, delta_norm, delta_1


# ── parameters ────────────────────────────────────────────────────────────────
L        = 100
N        = 10000
V_s      = 5
mu       = 6.6e-6
a2       = 0.1
theta    = 0.0
n_traits = 3
sigma_e2 = 1e-2

# ── 4 covariance-matrix cases ────────────────────────────────────────────────
cases = {
    'A': dict(diag_scale='full',      off_sign=+1, off_scale='full'),       # diag=σ²,  off=+σ²
    'B': dict(diag_scale='full',      off_sign=-1, off_scale='full'),       # diag=σ²,  off=-σ²
    'C': dict(diag_scale='per_trait', off_sign=+1, off_scale='per_trait'),  # diag=σ²/T, off=+σ²/T
    'D': dict(diag_scale='per_trait', off_sign=-1, off_scale='per_trait'),  # diag=σ²/T, off=-σ²/T
}

# also keep the baseline sigma_e2=0 run for reference (constant optimum)
print("Running sigma_e2 = 0 (baseline, constant optimum) ...")
zero_cov = np.zeros((n_traits, n_traits))
hist0, Vg0, _, _ = simulate_trajectory_nd(L, 0, N, V_s, mu, a2, theta, n_traits, zero_cov)
np.savetxt('hist_000_nd.txt',    hist0)
np.savetxt('Vg_hist_000_nd.txt', Vg0)

for label, cfg in cases.items():
    cov = make_cov_matrix(sigma_e2, n_traits, **cfg)
    print(f"\nCase {label}: cov matrix =\n{cov}")
    print(f"Running case {label} ...")

    hist, Vg, dn, d1 = simulate_trajectory_nd(L, sigma_e2, N, V_s, mu, a2,
                                              theta, n_traits, cov)
    np.savetxt(f'hist_{label}_nd.txt',       hist)
    np.savetxt(f'Vg_hist_{label}_nd.txt',    Vg)
    np.savetxt(f'delta_norm_{label}_nd.txt', dn)
    np.savetxt(f'delta_1_{label}_nd.txt',    d1)
    print(f"Saved hist_{label}_nd.txt, Vg_hist_{label}_nd.txt, "
          f"delta_norm_{label}_nd.txt, delta_1_{label}_nd.txt")

print(f"\nDone (n_traits={n_traits}).  Run figures_ndim.py to plot.")
