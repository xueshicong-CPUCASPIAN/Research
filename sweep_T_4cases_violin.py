# -*- coding: utf-8 -*-
"""
Vectorised sweep over T (number of trait dimensions) for the 4 covariance
cases.  THIS IS THE SINGLE SIMULATION SCRIPT: for each (T, case) it runs `rep`
replicate populations (vectorised over numpy arrays) and saves the per-locus
state (a_{1,l}^2 and p_l).  It produces the h^2 violin plot, and the
saved per-locus data file is also consumed by the histogram scripts
(sweep_T_4cases_hist.py, hist_a1sq_pq.py, *_summary.py), so the expensive
30000-generation simulation is run only once.

BURN-IN.  Populations start monomorphic, so the first few thousand generations are
a transient.  The first BURN_IN generations are discarded, then the state is
snapshotted every SAMPLE_EVERY generations and the snapshots are pooled along the
replicate axis (see the parameter block for the measured numbers behind the
choice).  Downstream scripts therefore receive rep*n_snap columns instead of rep
and need no change.  trace_Vg_over_gens_<tag>.pdf plots V_g from generation 0 with
the discarded burn-in shaded, so the choice can be verified against the data.

Each trait effect scales as a_t ~ sqrt(A/T), so E[||a||^2] = E[A] whatever T is.

CROSS-TERM RECORDING.  This script is also the only place the full effect matrix
a_{il} and the optimum displacement delta exist, so it additionally records, for ONE
tracked replicate per (T, case), the decomposition of the selection response

    dzbar_m/dt = (1/V_s) [ G_mm delta_m  +  sum_{l != m} G_ml delta_l ]
                           \___________/    \____________________/
                              OWN term          CROSS term

together with the per-locus kernel w_i = a_i . delta and the per-trait products
a_{im} delta_m.  None of this is recoverable from the snapshot data (a_{1,l}^2 loses
the sign, delta is never stored), which is why it has to be recorded here rather
than reconstructed later.  cross_term_figs.py turns it into figures without
re-simulating.

OUTPUT DIRECTORY.  Every file is written to OUTDIR (a dated results folder next to
this repository), not to the working directory.  Change RESULTS_DIR below to start a
new batch; all the plotting scripts read the same constant.

The per-trait DIRECTION distribution is selectable via `dir_dists`:
  'gauss' -- a_t ~ N(0, A/T)      (continuous magnitudes; ~8% of loci get Ns<1)
  'pm'    -- a_t = +-sqrt(A/T)    (the PAPER's |a_i| = a model; no neutral loci)
It is part of the output tag, so every file says which model produced it:
  tag = <A-dist>_<direction>_aT1_a2_<a2>   e.g. const_pm_aT1_a2_0.03

Output (one set per (A-dist, direction, a2)):
  hist_T_4cases_data_<tag>.npz  -- per-locus a_{1,l}^2 and p_l (read by hist scripts)
  Vg_sweep_T_4cases_<tag>.npz   -- final Vg per (case, T, replicate) (derived)
  violin_T_4cases_<tag>.pdf     -- violin plot, h^2 vs T, 4 cases side-by-side
  trace_Vg_over_gens_<tag>.pdf  -- V_g vs generation, burn-in shaded (diagnostic)
Plus one sigma^2=0 baseline pair per direction:
  Vg_baseline_sigma0_<direction>_a2_<a2>.npz
  hist_baseline_sigma0_<direction>_a2_<a2>.npz
Plus one cross-term file per (direction, T, case), read by cross_term_figs.py:
  cross_term_data_<direction>_T<T>_case<case>_a2_<a2>.npz
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
import time

# ── output directory ──────────────────────────────────────────────────────────
# All outputs (.npz and .pdf) go to a dated results folder next to this repository,
# so the repo stays code-only and each batch of runs is self-contained.  Every
# plotting script defines the same two lines, so changing the date here means
# changing it in all of them.
RESULTS_DIR = 'results Aug 10'
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', RESULTS_DIR)
os.makedirs(OUTDIR, exist_ok=True)
def out(name):  return os.path.join(OUTDIR, name)

# ── parameters ────────────────────────────────────────────────────────────────
L         = 100
N         = 10000
V_s       = 5
mu        = 6.6e-6
theta     = 0.0
sigma_e2  = 1e-3
maxiter   = 30000
rep       = 100            # replicates per (T, case)
# T=1 and T=2 are cheap and both matter to the cross-term figures: at T=1 the cross
# term must vanish identically (asserted below), and T=2 is the smallest T at which
# it exists, so it anchors the "cross vs T" curves.
T_list    = [1, 2, 5, 20, 100]

# ── burn-in and post-burn-in sampling ─────────────────────────────────────────
# The population starts monomorphic (p=0 everywhere), so the first few thousand
# generations are a transient in which V_g climbs to its mutation-selection-drift
# equilibrium.  Those generations must not enter any statistic.
#
# Numbers below come from a measured trajectory diagnostic (rep=20, 'pm', cases A
# and D plus the sigma^2=0 baseline, at T=1 and T=100, V_g recorded every 100
# generations from gen 0):
#   * slowest series reached 95% of its plateau by generation 4600 and 99% by 6500;
#   * block means over 5-10k, 10-15k, 15-20k, 20-25k, 25-30k showed no residual
#     trend (scatter ~5%, no direction), so the process is stationary from ~5000 on;
#   * the integrated autocorrelation time of V_g was 470-1300 generations, worst case
#     ~1300, so snapshots must be spaced well beyond that to be near-independent.
# BURN_IN = 10000 is ~1.5x the slowest 99% equilibration time; SAMPLE_EVERY = 2000
# is ~1.5x the worst autocorrelation time.  maxiter is unchanged, so the run costs
# the same as before but yields 11 snapshots instead of 1.
BURN_IN      = 10000   # generations discarded; no snapshot is taken before this
SAMPLE_EVERY = 2000    # generations between post-burn-in snapshots
TRACE_EVERY  = 100     # generations between V_g recordings for the trajectory plot

# snapshot generations: 10000, 12000, ..., 30000  -> 11 snapshots per replicate
SNAP_GENS = list(range(BURN_IN, maxiter + 1, SAMPLE_EVERY))
n_snap    = len(SNAP_GENS)
rep_eff   = rep * n_snap   # pooled sample size seen by every downstream script
assert BURN_IN < maxiter, "BURN_IN must leave at least one snapshot generation"

# ── cross-term recording (one tracked replicate per run) ──────────────────────
# The decomposition needs the signed effect matrix and delta at EVERY generation, so
# it cannot use the sparse snapshot grid above.  Recording it for all `rep` replicates
# would cost hundreds of MB per run, so it is recorded for replicate TRACK_REP only --
# the replicates are i.i.d., so one of them is a fair sample of the dynamics, and the
# aggregate statistics still come from all 100.
TRACK_REP       = 0     # which replicate is followed in detail
REC_EVERY       = 5     # generations between recordings of delta / own / cross / w
# The per-trait products a_{im} delta_m form an (L, T) array per recording, so they
# use a coarser grid and are kept only after burn-in (nothing plots them before it).
TRAIT_REC_EVERY = 100
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

# ── per-trait effect scaling a_t ~ N(0, A / T) ─────────────────────────────────
# Each trait effect shrinks as 1/T, so E[||a||^2] = E[A] independently of T.
# A1_TAG only names this scaling in filenames and .npz keys; it is kept so the
# already-generated *_aT1_*.npz data stay readable.
A1_TAG = 'aT1'

# ── per-trait DIRECTION distribution ──────────────────────────────────────────
# Given the scale A, each trait effect is  a_t = sqrt(A / T) * D_t,  where D_t is
# drawn here.  This is a separate axis from the A-scale `dists` above: `dists` sets
# how the overall magnitude varies between mutations, `dir_dists` sets the shape of
# each individual trait effect.
#   'gauss' : D_t ~ N(0,1)  -> a_t ~ N(0, A/T).  Continuous magnitudes; with
#             A = a2 constant this gives a_t^2 ~ (a2/T) * chi^2_1, so a sizeable
#             fraction of loci land at Ns < 1 and drift as if neutral.
#   'pm'    : D_t = +-1     -> a_t = +-sqrt(A/T).  Magnitude IDENTICAL at every
#             locus, only the sign is random.  This is the PAPER's single-trait
#             model ("|a_i| = a; a new allele is assigned a_i = +-a with equal
#             probability") generalised to T traits, and at T=1 with A = a2 it is
#             exactly a_i = +-sqrt(a2).  No locus is effectively neutral.
def draw_dir_gauss(n, T):  return np.random.normal(0, 1, size=(n, T))
def draw_dir_pm(n, T):     return np.random.choice([-1.0, 1.0], size=(n, T))

dir_dists = {
    'gauss': draw_dir_gauss,
    'pm':    draw_dir_pm,
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


def simulate_vec(T, cov, rep, draw_A, draw_dir):
    """Run `rep` replicates in parallel for `maxiter` generations.

    Returns
      a1_sq, p    -- per-locus focal-trait data, each (L, rep*n_snap).  The first
                     BURN_IN generations are discarded; the state is then snapshotted
                     every SAMPLE_EVERY generations and the snapshots are pooled along
                     the replicate axis, so downstream scripts see rep_eff = rep*n_snap
                     columns and need no change.  Snapshots are spaced past the
                     measured autocorrelation time, so they are near-independent.
      trace_gen   -- (n_trace,) generations at which the trajectory was recorded
      trace_Vg    -- (n_trace, rep) focal-trait V_g at those generations, recorded
                     from generation 0 so the burn-in itself stays inspectable.
      rec         -- dict of the full selection-response decomposition for replicate
                     TRACK_REP, recorded every REC_EVERY generations (see `record`
                     below for the contents and shapes).

    a1_sq = focal-trait squared effect a_{1,l}^2.  V_g(trait 1) is recovered as
    2 * sum_l a1_sq * p (1-p); returning the per-locus arrays (rather than just
    V_g) also lets the histogram scripts reuse this single simulation run.
    """
    # Effect scale A = a2 (constant; `draw_const`): the variable-scale distributions are
    # disabled, so A is the same for every mutation. Given A, each trait effect is
    # a_t = sqrt(A / T) * D_t with D_t from `draw_dir` ('gauss' -> N(0,1), 'pm' -> +-1),
    # drawn fresh per mutation, so E[||a||^2] = A independently of T.
    # effects starts empty (population monomorphic, p=0); each mutation fills its own
    # (locus, rep) row.
    effects = np.zeros((L, rep, T))
    opt = np.zeros((T, rep))
    p   = np.zeros((L, rep))
    Lchol = chol_or_svd(cov)

    snap_gens = set(SNAP_GENS)
    snaps_a1sq, snaps_p = [], []
    trace_gen, trace_Vg = [], []
    # cross-term records for replicate TRACK_REP (see the parameter block)
    rc_gen, rc_delta, rc_own, rc_cross, rc_Vg, rc_p, rc_w = [], [], [], [], [], [], []
    rc_tgen, rc_ad, rc_tp = [], [], []

    # The BLAS-backed `@` does not clear the FPU status word before it runs, so it
    # reports divide/overflow flags left behind by unrelated LAPACK calls (chol_or_svd
    # above) -- it fires even at t=0 on all-zero, all-finite arrays.  Those warnings are
    # suppressed here; the explicit finiteness check on pprime below is the real net.
    old_err = np.seterr(divide='ignore', over='ignore', invalid='ignore')

    for t in range(maxiter):
        fixed_loci_1 = (p == 1)
        p[fixed_loci_1] = 0
        opt = opt - 2 * np.einsum('lr,lrt->tr', fixed_loci_1.astype(float), effects)

        allele_expected = 2 * p
        zbar = np.einsum('lr,lrt->tr', allele_expected, effects)

        # ── cross-term recording, for replicate TRACK_REP ────────────────────
        # Done here, at the top of the loop, so the delta recorded at generation t is
        # exactly the delta that drives the selection step of generation t below.  (The
        # snapshot / trace recording at the bottom of the loop instead describes the
        # state AFTER the update, i.e. generation t+1 -- the two grids are labelled
        # accordingly and are not meant to line up.)
        # Cost is O(L*T) on one replicate; G is never formed.
        if t % REC_EVERY == 0 or (t >= BURN_IN and t % TRAIT_REC_EVERY == 0):
            eff0 = effects[:, TRACK_REP, :]                 # (L, T)
            p0   = p[:, TRACK_REP]                          # (L,)
            dlt0 = (opt - zbar)[:, TRACK_REP]               # (T,)
            if t % REC_EVERY == 0:
                pq0 = p0 * (1 - p0)
                w0  = eff0 @ dlt0                                    # (L,) a_i . delta
                Gd0 = 2 * ((w0 * pq0)[:, None] * eff0).sum(axis=0)   # (T,) (G delta)_m
                Gm0 = 2 * (eff0 ** 2 * pq0[:, None]).sum(axis=0)     # (T,) G_mm = V_g,m
                own0 = Gm0 * dlt0
                rc_gen.append(t)
                rc_delta.append(dlt0.copy())
                rc_own.append(own0)
                rc_cross.append(Gd0 - own0)
                rc_Vg.append(Gm0)
                rc_p.append(p0.copy())
                rc_w.append(w0)
            if t >= BURN_IN and t % TRAIT_REC_EVERY == 0:
                rc_tgen.append(t)
                rc_ad.append(eff0 * dlt0[None, :])          # (L, T) a_{im} delta_m
                rc_tp.append(p0.copy())

        # mutation: new alleles
        fixed_loci_0  = (p == 0)
        mutation_mask = (np.random.rand(L, rep) < N * mu) & fixed_loci_0
        np.place(p, mutation_mask, 1 / N)
        idx = np.where(mutation_mask)
        n_new = len(idx[0])
        # new mutation: scale A = a2 (constant), one per locus*rep event and SAME across
        # traits, plus a fresh direction drawn from `draw_dir`.
        A_new = draw_A(a2, T, n_new)                            # (n_new,): constant scale a2 per mutation
        effects[idx[0], idx[1], :] = (draw_dir(n_new, T)
                                      * np.sqrt(A_new / T)[:, None])

        # mutation at polymorphic loci
        poly_loci = np.logical_not(fixed_loci_0) & (p < 1 - 1 / N)
        p[poly_loci] += (
            (np.random.rand(np.sum(poly_loci)) < N * mu * (1 - p[poly_loci])) / N
            - (np.random.rand(np.sum(poly_loci)) < N * mu * p[poly_loci]) / N
        )

        # selection + drift
        pprime = p_prime_sel_opt(p, opt - zbar, effects, V_s)
        if not np.isfinite(pprime).all():
            np.seterr(**old_err)
            raise FloatingPointError(
                f'non-finite selection probability at generation {t} (T={T})')
        p = np.random.binomial(N, pprime) / N

        # optimum shift via cholesky factor (one sample per replicate)
        z = np.random.randn(T, rep)
        opt = (1 - theta) * opt + Lchol @ z

        # ── recording.  `t` counts completed updates, so the state now describes
        #    generation t+1. ────────────────────────────────────────────────────
        gen = t + 1
        if gen % TRACE_EVERY == 0:
            a1_sq_now = effects[:, :, 0] ** 2
            trace_gen.append(gen)
            trace_Vg.append(2 * np.sum(a1_sq_now * p * (1 - p), axis=0))
        if gen in snap_gens:                     # post-burn-in snapshot
            snaps_a1sq.append(effects[:, :, 0] ** 2)
            snaps_p.append(p.copy())

    np.seterr(**old_err)

    # Pool the snapshots along the replicate axis; V_g is recovered downstream as
    # 2 * sum_l a1_sq * p (1-p), which also feeds the histogram scripts.
    a1_sq = np.concatenate(snaps_a1sq, axis=1)   # (L, rep*n_snap)
    p_out = np.concatenate(snaps_p,    axis=1)   # (L, rep*n_snap)

    # The per-locus arrays (p, w, ad) dominate the file size and are only ever plotted,
    # so they are stored single precision; the per-trait series stay float64 because the
    # T=1 cross-term check below compares them at the 1e-16 level.
    rec = dict(
        gen   = np.array(rc_gen),                            # (n_rec,)
        delta = np.array(rc_delta),                          # (n_rec, T)
        own   = np.array(rc_own),                            # (n_rec, T) G_mm delta_m
        cross = np.array(rc_cross),                          # (n_rec, T) sum_{l!=m} G_ml delta_l
        Vg    = np.array(rc_Vg),                             # (n_rec, T) G_mm
        p     = np.array(rc_p,  dtype=np.float32),           # (n_rec, L)
        w     = np.array(rc_w,  dtype=np.float32),           # (n_rec, L) a_i . delta
        tgen  = np.array(rc_tgen),                           # (n_tr,)
        ad    = np.array(rc_ad, dtype=np.float32),           # (n_tr, L, T) a_{im} delta_m
        tp    = np.array(rc_tp, dtype=np.float32),           # (n_tr, L)
    )
    return a1_sq, p_out, np.array(trace_gen), np.stack(trace_Vg, axis=0), rec


# ── main loop ────────────────────────────────────────────────────────────────
cases = {
    'A': dict(diag_scale='full',      off_sign=+1, off_scale='full'),
    'B': dict(diag_scale='full',      off_sign=-1, off_scale='full'),
    'C': dict(diag_scale='per_trait', off_sign=+1, off_scale='per_trait'),
    'D': dict(diag_scale='per_trait', off_sign=-1, off_scale='per_trait'),
}

# ── σ²=0 baseline (static optimum): denominator for the V_g ratio plot ─────────
# At σ²=0 the optimum never moves, so the covariance matrix is all zeros and the
# four cases A–D coincide. We therefore run just one simulation per T using
# draw_const (A ≡ a2), and save the per-replicate focal-trait V_g. This is
# the *simulated* static-optimum baseline, to compare against the analytic
# Latter–Bulmer value 4·L·μ·V_s in plot_Vg_ratio_over_T.py.
print("\n############## σ²=0 baseline (static optimum, cases collapse) ##############")
# (dir_name, T) -> (generations, mean V_g) so the per-(dist,dir) trace figure below
# can show the baseline transient next to the four cases.
baseline_traces = {}
# (dir_name, a2, T) -> the tracked-replicate record of the sigma^2=0 run.  The cross-term
# figures draw it as the black reference curve in every case panel, so it is kept in
# memory here and written into each case's cross_term_data_*.npz below.
baseline_cross = {}
# One baseline per direction distribution: a 'pm' run must be compared against a
# 'pm' baseline, so the direction name goes into the baseline filenames too.
for dir_name, draw_dir in dir_dists.items():
    for a2 in a2_values:                   # sets the global a2 read by simulate_vec
        base    = {'T_list': np.array(T_list)}   # totals    -> Vg_baseline_sigma0_*.npz
        base_pl = {'T_list': np.array(T_list)}   # per-locus -> hist_baseline_sigma0_*.npz
        Vg_T = np.zeros((len(T_list), rep_eff))
        for ti, T in enumerate(T_list):
            t0 = time.time()
            cov0 = make_cov_matrix(0.0, T, diag_scale='full',
                                   off_sign=+1, off_scale='full')   # all zeros
            a1_sq, p, tr_gen, tr_Vg, rec0 = simulate_vec(T, cov0, rep, draw_const, draw_dir)
            Vg_T[ti] = 2 * np.sum(a1_sq * p * (1 - p), axis=0)
            # keep the mean trajectory so the main loop can overlay it on the trace figure
            baseline_traces[(dir_name, T)] = (tr_gen, tr_Vg.mean(axis=1))
            baseline_cross[(dir_name, float(a2), T)] = rec0
            # keep per-locus arrays so the rank/hist scripts can overlay the baseline
            base_pl[f'{A1_TAG}_T{T}_a1sq'] = a1_sq
            base_pl[f'{A1_TAG}_T{T}_p']    = p
            print(f"  baseline dir={dir_name}  T={T}: "
                  f"mean Vg = {Vg_T[ti].mean():.5g}  [{time.time()-t0:.1f}s]")
        base[A1_TAG] = Vg_T
        np.savez(out(f'Vg_baseline_sigma0_{dir_name}_a2_{a2:.2f}.npz'), **base)
        np.savez(out(f'hist_baseline_sigma0_{dir_name}_a2_{a2:.2f}.npz'), **base_pl)
        print(f"Saved Vg_baseline_sigma0_{dir_name}_a2_{a2:.2f}.npz and "
              f"hist_baseline_sigma0_{dir_name}_a2_{a2:.2f}.npz")

for (dist_name, draw_A), (dir_name, draw_dir), a2 in itertools.product(
        dists.items(), dir_dists.items(), a2_values):
    tag = f"{dist_name}_{dir_name}_{A1_TAG}_a2_{a2:.2f}"
    print(f"\n############## dist = {dist_name}  dir = {dir_name}  "
          f"a2 = {a2:.3f}  ({tag}) ##############")

    # results[case][T_idx] = Vg array of length rep_eff (replicates x snapshots)
    results = {label: np.zeros((len(T_list), rep_eff)) for label in cases}
    # per-locus arrays saved for the histogram scripts (read by sweep_T_4cases_hist.py etc.)
    save_dict = {'T_list': np.array(T_list)}
    traces = {}          # (T, case) -> (generations, mean V_g) for the trace figure

    for ti, T in enumerate(T_list):
        print(f"\n=== T = {T} ===")
        for label, cfg in cases.items():
            t0 = time.time()
            cov = make_cov_matrix(sigma_e2, T, **cfg)
            a1_sq, p, tr_gen, tr_Vg, rec = simulate_vec(T, cov, rep, draw_A, draw_dir)
            # focal-trait V_g per pooled sample = 2 * sum_l a_{1,l}^2 p_l(1-p_l)
            Vg = 2 * np.sum(a1_sq * p * (1 - p), axis=0)
            results[label][ti] = Vg
            traces[(T, label)] = (tr_gen, tr_Vg.mean(axis=1))
            # stash per-locus arrays so the histogram scripts can reuse this run
            save_dict[f'{label}_T{T}_a1sq'] = a1_sq
            save_dict[f'{label}_T{T}_p']    = p
            h2 = Vg / (1 + Vg)
            print(f"  Case {label}: mean h² = {h2.mean():.4f}  std = {h2.std():.4f}  "
                  f"[{time.time()-t0:.1f}s]")

            # ── cross-term file, read by cross_term_figs.py ──────────────────
            if T == 1:
                # Correctness check: a single trait has no l != m, so the cross term
                # must vanish.  It is not bitwise zero -- (G delta)_1 and G_11 delta_1
                # multiply by delta_1 at different points in the summation, so they
                # differ by floating-point round-off (a few eps).
                nz = rec['own'] != 0
                relmax = np.abs(rec['cross'][nz] / rec['own'][nz]).max() if nz.any() else 0.0
                assert relmax < 1e-12, f'cross term is nonzero at T=1 (relative {relmax:.3g})'
                print(f"    T=1 check passed: cross term vanishes "
                      f"(max relative residual {relmax:.2g}, round-off only)")
            m_post   = rec['gen'] >= BURN_IN
            rms_own  = np.sqrt(np.mean(rec['own'][m_post, 0] ** 2))
            rms_cros = np.sqrt(np.mean(rec['cross'][m_post, 0] ** 2))
            print(f"    cross term (trait 1, rep {TRACK_REP}, post burn-in): "
                  f"RMS own = {rms_own:.4g}, RMS cross = {rms_cros:.4g}, "
                  f"ratio = {rms_cros / rms_own:.3f}")
            rec0 = baseline_cross[(dir_name, float(a2), T)]
            ctag = f'{dir_name}_T{T}_case{label}_a2_{a2:.2f}'
            np.savez(out(f'cross_term_data_{ctag}.npz'),
                     gen=rec['gen'], delta=rec['delta'], own=rec['own'],
                     cross=rec['cross'], Vg=rec['Vg'], p=rec['p'], w=rec['w'],
                     tgen=rec['tgen'], ad=rec['ad'], tp=rec['tp'],
                     gen0=rec0['gen'], delta0=rec0['delta'], own0=rec0['own'],
                     cross0=rec0['cross'], Vg0=rec0['Vg'], p0=rec0['p'], w0=rec0['w'],
                     T=T, L=L, N=N, V_s=V_s, a2=a2, sigma_e2=sigma_e2, rep=rep,
                     BURN_IN=BURN_IN, REC_EVERY=REC_EVERY,
                     TRAIT_REC_EVERY=TRAIT_REC_EVERY, TRACK_REP=TRACK_REP,
                     dir_name=dir_name, case=label)
            print(f"    Saved cross_term_data_{ctag}.npz")

    # ── save ──────────────────────────────────────────────────────────────────
    # (1) per-locus data — consumed by sweep_T_4cases_hist.py, hist_a1sq_pq.py, *_summary.py
    np.savez(out(f'hist_T_4cases_data_{tag}.npz'), **save_dict)
    print(f"\nSaved hist_T_4cases_data_{tag}.npz")
    # (2) derived per-sample Vg (convenience / downstream)
    np.savez(out(f'Vg_sweep_T_4cases_{tag}.npz'),
             T_list=np.array(T_list),
             A=results['A'], B=results['B'], C=results['C'], D=results['D'])
    print(f"Saved Vg_sweep_T_4cases_{tag}.npz")

    # ── V_g trajectory over generations (burn-in diagnostic) ─────────────────
    # One panel per T; each curve is the across-replicate mean V_g.  The discarded
    # burn-in is shaded and the snapshot generations are marked, so the choice of
    # BURN_IN can be checked against the data rather than taken on trust.
    tr_colors = {'A': 'C0', 'B': 'C3', 'C': 'C2', 'D': 'C1'}
    ncol = 2
    nrow = int(np.ceil(len(T_list) / ncol))
    figt, axest = plt.subplots(nrow, ncol, figsize=(7.0 * ncol, 4.2 * nrow),
                               squeeze=False)
    for ti, T in enumerate(T_list):
        axt = axest.ravel()[ti]
        axt.axvspan(0, BURN_IN, color='0.85', zorder=0)
        axt.text(BURN_IN / 2, 0.97, f'burn-in\n(discarded)\n{BURN_IN} gens',
                 transform=axt.get_xaxis_transform(), ha='center', va='top',
                 fontsize=8, color='0.35')
        for label in cases:
            g, v = traces[(T, label)]
            axt.plot(g, v, color=tr_colors[label], lw=1.2, label=f'case {label}')
        if (dir_name, T) in baseline_traces:
            g, v = baseline_traces[(dir_name, T)]
            axt.plot(g, v, color='k', lw=1.4, label=r'$\sigma^2=0$ baseline')
        for sg in SNAP_GENS:
            axt.axvline(sg, color='0.4', ls=':', lw=0.6, zorder=0)
        axt.set_yscale('log')
        axt.set_xlim(0, maxiter)
        axt.set_title(f'T = {T}', fontsize=11)
        axt.set_xlabel('generation')
        axt.set_ylabel(r'$V_g$(trait 1), mean over replicates')
        axt.grid(True, which='both', alpha=0.3)
        if ti == 0:
            axt.legend(fontsize=8, loc='lower right')
    for pi in range(len(T_list), nrow * ncol):
        axest.ravel()[pi].axis('off')
    figt.suptitle(
        rf'$V_g$ vs generation  (A~{dist_name}, dir={dir_name}, $a^2$={a2:.2f})'
        f'\nburn-in = {BURN_IN} gens discarded; {n_snap} snapshots every '
        f'{SAMPLE_EVERY} gens (dotted) -> {rep_eff} pooled samples per (T, case)',
        fontsize=11)
    figt.tight_layout(rect=[0, 0, 1, 0.93])
    figt.savefig(out(f'trace_Vg_over_gens_{tag}.pdf'), bbox_inches='tight')
    plt.close(figt)
    print(f"Saved trace_Vg_over_gens_{tag}.pdf")

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
    dir_label = {'pm': r'$a_t=\pm\sqrt{A/T}$ (paper)',
                 'gauss': r'$a_t\sim N(0,A/T)$'}.get(dir_name, dir_name)
    ax.set_title(r'Violin plot of $h^2$ across replicates ' +
                 f'(A~{dist_name}, dir={dir_name}: {dir_label}, '
                 f'{rep} reps x {n_snap} post-burn-in snapshots = {rep_eff} samples, '
                 f'$a^2={a2:.2f}$, $\\sigma^2=10^{{-3}}$, $V_s=5$, $N=10^4$, $L=100$)',
                 fontsize=11)
    ax.set_ylim([y_lo, y_hi])
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(out(f'violin_T_4cases_{tag}.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"Saved violin_T_4cases_{tag}.pdf")
