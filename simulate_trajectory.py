# -*- coding: utf-8 -*-
"""
Runs a single-replicate trajectory simulation to produce the history files
needed for Figure 1 in figures.py:
  hist_000.txt       -- allele frequency trajectories, sigma_e2 = 0
  hist_001.txt       -- allele frequency trajectories, sigma_e2 = 1e-2
  delta_hist_001.txt -- delta_t = opt - zbar trajectory, sigma_e2 = 1e-2

Shape of hist files:   (maxiter, L)  -- each row is one generation
Shape of delta file:   (maxiter,)
"""

import numpy as np

# ── shared functions (same as simulate_mpi.py) ────────────────────────────────
def pmap(rho):
    return rho / (1 + rho)

def rhomap(p):
    return p / (1 - p)

def p_prime_sel_opt(p, delt_opt, gam, sign, V_s):
    S = 1 / (2 * V_s)
    p = pmap(rhomap(p) * np.exp(2 * S * gam * sign * (delt_opt + 0.5 * gam * sign * (2 * p - 1))))
    return p

# ── trajectory simulation (single replicate) ─────────────────────────────────
def simulate_trajectory(L, sigma_e2, N, V_s, mu, a2, theta):
    """
    Returns:
        hist  : (maxiter, L) array of allele frequencies at every generation
        delta : (maxiter,)   array of opt - zbar at every generation
    """
    a = np.sqrt(a2)
    sign = 2 * np.random.randint(0, 2, L) - 1   # shape (L,)
    opt = 0.0
    p = np.zeros(L)
    maxiter = int(10 * N)

    hist  = np.zeros((maxiter, L))
    delta = np.zeros(maxiter)

    for t in range(maxiter):
        if t % 1000 == 0:
            print(f"  t = {t} / {maxiter}")

        # fix loci that reached p=1
        fixed_loci_1 = (p == 1)
        p[fixed_loci_1] = 0
        opt = opt - 2 * a * np.sum(sign * fixed_loci_1)

        zbar = np.sum(2 * a * sign * p**2 + a * sign * p * (1 - p))

        # record state BEFORE selection step
        hist[t, :]  = p
        delta[t]    = opt - zbar

        # mutation: new alleles enter at freq 1/N
        fixed_loci_0 = (p == 0)
        mutation_mask = (np.random.rand(L) < N * mu) & fixed_loci_0
        p[mutation_mask] = 1 / N
        sign[mutation_mask] = 2 * np.random.randint(0, 2, np.sum(mutation_mask)) - 1

        # mutation at polymorphic loci
        poly_loci = np.logical_not(fixed_loci_0) & (p < 1 - 1 / N)
        p[poly_loci] += (
            (np.random.rand(np.sum(poly_loci)) < N * mu * (1 - p[poly_loci])) / N
            - (np.random.rand(np.sum(poly_loci)) < N * mu * p[poly_loci]) / N
        )

        # selection + drift
        p = np.random.binomial(N, p_prime_sel_opt(p, opt - zbar, a, sign, V_s)) / N

        # optimum shift
        if sigma_e2 > 0:
            opt = (1 - theta) * opt + np.random.normal(0, np.sqrt(sigma_e2))
        # sigma_e2 == 0: opt stays at 0

    return hist, delta


# ── parameters (match supervisor's Figure 1) ─────────────────────────────────
L      = 100
N      = 10000
V_s    = 5
mu     = 6.6e-6
a2     = 0.1
theta  = 0.0

# ── run sigma_e2 = 0  →  hist_000.txt ────────────────────────────────────────
print("Running sigma_e2 = 0 ...")
hist0, _ = simulate_trajectory(L, sigma_e2=0, N=N, V_s=V_s, mu=mu, a2=a2, theta=theta)
np.savetxt('hist_000.txt', hist0)
print("Saved hist_000.txt")

# ── run sigma_e2 = 1e-2  →  hist_001.txt + delta_hist_001.txt ────────────────
print("Running sigma_e2 = 1e-2 ...")
hist1, delta1 = simulate_trajectory(L, sigma_e2=1e-2, N=N, V_s=V_s, mu=mu, a2=a2, theta=theta)
np.savetxt('hist_001.txt',       hist1)
np.savetxt('delta_hist_001.txt', delta1)
print("Saved hist_001.txt and delta_hist_001.txt")

print("Done. You can now run figures.py to produce Figure 1.")
