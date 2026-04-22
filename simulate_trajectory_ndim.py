# -*- coding: utf-8 -*-
"""
n-dimensional version of simulate_trajectory.py.
Runs two single-replicate simulations and saves history files for figures_ndim.py:

  hist_000_nd.txt        -- allele frequency trajectories, sigma_e2=0,    shape (maxiter, L)
  hist_001_nd.txt        -- allele frequency trajectories, sigma_e2=1e-2,  shape (maxiter, L)
  Vg_hist_000_nd.txt     -- genetic variance per generation, sigma_e2=0,   shape (maxiter,)
  Vg_hist_001_nd.txt     -- genetic variance per generation, sigma_e2=1e-2,shape (maxiter,)
  delta_norm_001_nd.txt  -- ||opt - zbar||_2 per generation, sigma_e2=1e-2,shape (maxiter,)
"""

import numpy as np

# ── shared functions (same as simulate_mpi n_dimension.py) ───────────────────
def pmap(rho):
    return rho / (1 + rho)

def rhomap(p):
    return p / (1 - p)

def p_prime_sel_opt(p, delt_opt, effects, V_s):
    """
    p:        (L,)
    delt_opt: (n_traits,)
    effects:  (L, n_traits)
    """
    S = 1 / (2 * V_s)
    dot_term = np.einsum('lt,t->l', effects, delt_opt)   # (L,)
    norm2    = np.sum(effects**2, axis=1)                 # (L,)
    expo = 2 * S * (dot_term + 0.5 * norm2 * (2 * p - 1))
    return pmap(rhomap(p) * np.exp(expo))

# ── trajectory simulation (single replicate, n-D) ────────────────────────────
def simulate_trajectory_nd(L, sigma_e2, N, V_s, mu, a2, theta, n_traits):
    """
    Returns:
        hist       : (maxiter, L)   allele frequencies at every generation
        Vg_hist    : (maxiter,)     genetic variance at every generation
        delta_norm : (maxiter,)     ||opt - zbar||_2 at every generation
    """
    a = np.sqrt(a2)
    effects = np.random.normal(0, a, size=(L, n_traits))  # (L, n_traits)
    opt = np.zeros(n_traits)
    p   = np.zeros(L)
    maxiter = int(10 * N)

    hist       = np.zeros((maxiter, L))
    Vg_hist    = np.zeros(maxiter)
    delta_norm = np.zeros(maxiter)

    for t in range(maxiter):
        if t % 1000 == 0:
            print(f"  t = {t} / {maxiter}")

        # fix loci that reached p=1
        fixed_loci_1 = (p == 1)
        p[fixed_loci_1] = 0
        opt = opt - 2 * np.einsum('l,lt->t', fixed_loci_1.astype(float), effects)

        allele_expected = 2 * p**2 + 2 * p * (1 - p)          # (L,)  = 2p
        zbar = np.einsum('l,lt->t', allele_expected, effects)  # (n_traits,)

        delt = opt - zbar  # (n_traits,)

        # record state
        hist[t, :]    = p
        Vg_hist[t]    = 2 * np.sum(np.sum(effects**2, axis=1) * p * (1 - p)) / n_traits
        # divide by n_traits to get per-trait average Vg
        delta_norm[t] = np.linalg.norm(delt)

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

        # optimum shift
        if sigma_e2 > 0:
            opt = (1 - theta) * opt + np.random.normal(0, np.sqrt(sigma_e2/n_traits), size=n_traits)
            # divide by n_traits so total ||delta_opt||^2 variance = sigma_e2 regardless of n_traits

    return hist, Vg_hist, delta_norm


# ── parameters ────────────────────────────────────────────────────────────────
L       = 100
N       = 10000
V_s     = 5
mu      = 6.6e-6
a2      = 0.1
theta   = 0.0
n_traits = 3   # number of trait dimensions

# ── run sigma_e2 = 0 ──────────────────────────────────────────────────────────
print("Running sigma_e2 = 0 ...")
hist0, Vg0, _ = simulate_trajectory_nd(L, sigma_e2=0, N=N, V_s=V_s,
                                        mu=mu, a2=a2, theta=theta, n_traits=n_traits)
np.savetxt('hist_000_nd.txt',    hist0)
np.savetxt('Vg_hist_000_nd.txt', Vg0)
print("Saved hist_000_nd.txt, Vg_hist_000_nd.txt")

# ── run sigma_e2 = 1e-2 ───────────────────────────────────────────────────────
print("Running sigma_e2 = 1e-2 ...")
hist1, Vg1, delta1 = simulate_trajectory_nd(L, sigma_e2=1e-2, N=N, V_s=V_s,
                                             mu=mu, a2=a2, theta=theta, n_traits=n_traits)
np.savetxt('hist_001_nd.txt',       hist1)
np.savetxt('Vg_hist_001_nd.txt',    Vg1)
np.savetxt('delta_norm_001_nd.txt', delta1)
print("Saved hist_001_nd.txt, Vg_hist_001_nd.txt, delta_norm_001_nd.txt")

print(f"Done (n_traits={n_traits}). Run figures_ndim.py to plot.")
