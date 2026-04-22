# Parameters & Variables Reference

Complete annotated reference for all parameters and variables used across
`simulate_mpi.py`, `simulate_mpi n_dimension.py`, `simulate_trajectory.py`,
and `simulate_trajectory_ndim.py`.

---

## 1. Biological Model Parameters

These are the top-level inputs that define the evolutionary scenario.
They are set in the parameter-sweep arrays at the bottom of each script
and passed into `simulate()` / `simulate_trajectory()` as a tuple.

---

### `L` — Number of Loci

- **Symbol:** L
- **Typical value:** 100
- **Type:** integer
- **Meaning:** The number of independent genetic loci (sites in the genome)
  that contribute additively to the trait. Each locus segregates a single
  biallelic variant (wild-type vs. mutant). More loci → more potential
  sources of genetic variance, and a higher mutation supply L·μ.
- **Where used:** shapes of `p`, `effects`, `sign`, `fixed_loci_0/1`,
  `mutation_mask`, `poly_loci`, `hist`.

---

### `N` — Population Size

- **Symbol:** N
- **Typical value:** 10 000
- **Type:** integer
- **Meaning:** Haploid census population size. Because individuals are
  treated as diploid for phenotype purposes (2p² for AA, p(1-p) for Aa)
  but the Wright-Fisher sampling uses N, N here is effectively the number
  of diploid individuals. Genetic drift strength scales as 1/N — larger N
  means drift is weaker. Also controls simulation length: `maxiter = 10·N`.
- **Where used:** `maxiter`, `mutation_mask` threshold (`N·μ`), binomial
  sampling in `np.random.binomial(N, p_prime, ...)`, mutation at
  polymorphic loci.

---

### `V_s` — Stabilizing Selection Strength

- **Symbol:** V_s (also written V_s in the literature)
- **Typical values:** 5, 20
- **Type:** float
- **Meaning:** The variance parameter of the Gaussian fitness function
  W(z) ∝ exp(−(z−z*)² / 2V_s). **A larger V_s means weaker selection**
  (a flatter, broader fitness peak). The selection coefficient used
  internally is S = 1/(2·V_s). The Latter-Bulmer prediction for baseline
  genetic variance is V_g^LB = 4·L·μ·V_s, so stronger selection (smaller
  V_s) reduces equilibrium V_g.
- **Where used:** `p_prime_sel_opt` via `S = 1/(2*V_s)`.

---

### `mu` — Mutation Rate

- **Symbol:** μ
- **Typical value:** 6.6×10⁻⁶
- **Type:** float
- **Meaning:** Per-locus, per-generation probability that a new mutant
  allele arises. Controls mutation supply: the expected number of new
  mutations entering the population per locus per generation is N·μ (used
  as a Poisson/Bernoulli rate in the mutation step). Also governs
  back-mutation at polymorphic loci.
- **Where used:** `mutation_mask`, the polymorphic-loci mutation update.

---

### `a2` — Effect-Size Variance

- **Symbol:** a² (input to simulation)
- **Typical value:** 0.1
- **Type:** float
- **Meaning:** Variance of the distribution from which new mutational
  effect sizes are drawn. Each new allele's effect is sampled as
  N(0, √a2) per trait component. Controls how large a phenotypic step
  a single mutation can cause. Larger a2 → bigger per-mutation effects →
  fewer segregating alleles needed to maintain a given V_g.
- **Where used:** `a = np.sqrt(a2)`, all draws of `effects`.

---

### `a` — Effect-Size Standard Deviation

- **Symbol:** a
- **Computed as:** `a = np.sqrt(a2)`
- **Meaning:** Standard deviation of mutational effect sizes. In the 1D
  model each allele shifts the trait by ±a (sign chosen randomly). In
  the n-D model each allele has an effect *vector* whose components are
  each drawn independently from N(0, a).
- **Where used:** `np.random.normal(0, a, ...)` to initialise and refresh
  `effects`; in the 1D `zbar` formula.

---

### `sigma_e2` — Environmental Fluctuation Intensity

- **Symbol:** σ²
- **Typical values:** 0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2
- **Type:** float
- **Meaning:** Variance of the per-generation random shift in the trait
  optimum. σ² = 0 means a static optimum (pure stabilising selection,
  Latter-Bulmer regime). σ² > 0 introduces a moving optimum that forces
  the population to continually track it, increasing V_g above the
  Latter-Bulmer baseline. This is the primary experimental variable in
  the paper.

  - **1D:** each generation `opt += N(0, √σ²)`
  - **n-D:** each component gets `N(0, √(σ²/T))` so the total squared
    displacement has expectation σ² regardless of T.

- **Where used:** optimum-shift step at end of each generation.

---

### `theta` — Ornstein-Uhlenbeck Restoring Force

- **Symbol:** θ
- **Typical value:** 0.0
- **Type:** float (range [0, 1])
- **Meaning:** Controls mean-reversion of the optimum. Each generation:

      opt_new = (1 − θ)·opt + ε,    ε ~ N(0, √(σ²/T))

  - θ = 0: pure Brownian motion — optimum drifts without any pull back.
  - θ = 1: optimum reset to 0 every generation (maximum reversion).
  - 0 < θ < 1: OU process, optimum fluctuates around 0 with stationary
    variance σ²/(2θ − θ²) ≈ σ²/(2θ).

  Currently kept at 0 in all sweep runs; included for future experiments.
- **Where used:** optimum-shift step: `opt = (1 - theta)*opt + noise`.

---

### `n_traits` — Number of Trait Dimensions

- **Symbol:** T
- **Typical values (sweep):** 1, 2, 3, 5
- **Type:** integer
- **Meaning:** The number of phenotypic traits under simultaneous
  stabilising + directional selection (n-D extension only). Each locus
  now has an *effect vector* of length T rather than a scalar effect.
  Increasing T spreads each mutation's effect across more dimensions,
  so on average each individual mutation is less effective at moving the
  population toward the (scalar-equivalent) optimum. This is the key
  variable for testing dimensionality effects on V_g.
- **Where used:** shape of `effects` (axis 2), shape of `opt` and `zbar`
  (axis 0), normalisation of σ²/n_traits, normalisation of returned V_g.

---

### `rep` / `all_reps` — Number of Replicate Populations

- **`all_reps`:** Total replicates (default 100). Set at top of MPI script.
- **`rep` / `rep_local`:** `all_reps / size` — replicates handled by each
  MPI core.
- **Meaning:** Independent replicate populations evolved under identical
  parameters. Their final V_g values form the distribution plotted in
  violin plots and compared to theory. More replicates → tighter estimate
  of the mean and variance of V_g.

---

## 2. State Variables (evolve each generation)

---

### `p` — Allele Frequencies

- **Shape (MPI):** (L, rep)   |   **Shape (trajectory):** (L,)
- **Range:** [0, 1]
- **Meaning:** Frequency of the mutant allele at each locus in each
  replicate population. The entire evolutionary dynamics of the model are
  summarised in how `p` changes each generation through four processes:
  fixation handling → mutation → selection → drift.
  - p = 0: allele absent
  - p = 1: allele fixed (treated as new wild-type and reset to 0)
  - 0 < p < 1: polymorphic, contributes to V_g

---

### `opt` — Trait Optimum

- **Shape (1D MPI):** (rep,)
- **Shape (n-D MPI):** (n_traits, rep)
- **Shape (trajectory):** scalar (1D) or (n_traits,) (n-D)
- **Meaning:** The current phenotypic optimum — the trait value with
  highest fitness. Initialised at 0. Each generation it shifts by a
  Gaussian random step (controlled by σ² and θ). When alleles fix, `opt`
  is re-centred to absorb their contribution (avoids `opt` growing
  without bound).

---

### `effects` — Allelic Effect Vectors

- **Shape (n-D MPI):** (L, rep, n_traits)
- **Shape (n-D trajectory):** (L, n_traits)
- **Meaning:** The phenotypic effect vector of the mutant allele at each
  locus. `effects[l, r, :]` is the T-dimensional vector by which the
  mean phenotype changes if locus l in replicate r goes from 0 copies to
  1 additional copy of the mutant. Components drawn independently from
  N(0, a). Re-drawn when a new mutation enters a locus.
- **1D equivalent:** `a * sign` (separate `a` scalar and `sign` array).

---

### `sign` — Allelic Effect Direction (1D only)

- **Shape:** (L, rep)
- **Values:** +1 or −1
- **Meaning:** 1D model only. Each allele either increases (+1) or
  decreases (−1) the trait by amount `a`. Replaced in the n-D model by
  the full `effects` vector, which encodes both direction and magnitude
  per trait.

---

### `zbar` — Population Mean Phenotype

- **Shape (1D):** (rep,) scalar per replicate
- **Shape (n-D):** (n_traits, rep)
- **Computed as:**

      allele_expected = 2p²  +  2p(1−p)  =  2p        (expected copies per individual)
      zbar = Σ_l  allele_expected[l] · effects[l]

- **Meaning:** The mean phenotype of the population. The factor 2p comes
  from diploid genetics: an AA individual contributes 2 copies, Aa
  contributes 1 copy (expected), aa contributes 0 copies of the mutant
  allele. The mean contribution of locus l to the trait is thus
  2p_l · a_l (in n-D: 2p_l · effects[l]).

---

### `delt` / `opt - zbar` — Displacement from Optimum

- **Shape (1D):** (rep,)
- **Shape (n-D):** (n_traits, rep)
- **Meaning:** The gap between the current optimum and the population
  mean phenotype. This is the driver of directional selection: if the
  population lags behind the optimum, alleles whose effects point toward
  the optimum are selectively favoured. Its norm ||δ|| is recorded in
  `delta_norm` for trajectory plots.

---

## 3. Derived / Intermediate Variables

---

### `S` — Selection Coefficient

- **Computed as:** `S = 1 / (2 * V_s)`
- **Meaning:** Compact form of the selection strength used in the
  log-odds update formula. Inverse of 2·V_s.

---

### `rhomap(p)` — Log-Odds Transform

- **Formula:** `p / (1 − p)`
- **Meaning:** Maps allele frequency to odds ratio. Used to apply
  multiplicative selection in log-odds space, which keeps allele
  frequencies in [0,1] without clamping.

---

### `pmap(rho)` — Inverse Log-Odds

- **Formula:** `rho / (1 + rho)`
- **Meaning:** Maps log-odds back to probability. Together with
  `rhomap`, these two functions implement the logistic/sigmoid trick
  for updating allele frequencies under selection.

---

### `p_prime_sel_opt(p, delt_opt, effects, V_s)` — Post-Selection Frequency

- **Returns:** array same shape as `p`
- **Formula:**

      expo   = 2S · (dot_term  +  0.5 · norm2 · (2p − 1))
      p_new  = pmap( rhomap(p) · exp(expo) )

- **Meaning:** The allele frequency after one generation of selection,
  before genetic drift. The exponent has two biological terms:
  - **`dot_term`** = **a**_l · **δ** = directional selection.
    Allele l is favoured if its effect vector points in the same
    direction as the displacement δ = opt − zbar (i.e. toward the
    optimum).
  - **`0.5 · norm2 · (2p − 1)`** = stabilising selection.
    This term penalises large-effect alleles at extreme frequencies
    (pure stabilising selection, independent of optimum displacement).
    It creates heterozygote advantage.

---

### `dot_term` — Directional Selection Inner Product

- **Computed as:** `np.einsum('lrt,tr->lr', effects, delt_opt)`
- **Shape:** (L, rep)
- **Meaning:** For each locus l and replicate r, the dot product of the
  allele's effect vector with the displacement vector δ. A positive value
  means the allele "helps" the population move toward the optimum.

---

### `norm2` — Squared Magnitude of Effect Vector

- **Computed as:** `np.sum(effects**2, axis=2)` (n-D)
- **Shape:** (L, rep)
- **Meaning:** ‖**a**_l‖² — the total squared phenotypic effect of allele
  l summed across all T traits. Appears in the stabilising selection term
  of the log-odds update. Larger-effect alleles face stronger stabilising
  selection.

---

### `allele_expected` — Expected Allele Copies per Individual

- **Computed as:** `2*p**2 + 2*p*(1-p)` = `2p`
- **Meaning:** For a diploid individual with mutant frequency p:
  - P(AA) = p² → contributes 2 copies
  - P(Aa) = 2p(1−p) → contributes 1 copy
  - P(aa) = (1−p)² → contributes 0 copies
  Expected copies = 2p² + 2·p(1−p)·1 = 2p.
  Used to compute `zbar`.

---

### `fixed_loci_1` — Mask: Loci Fixed for Mutant

- **Shape:** (L, rep) boolean
- **Meaning:** True where p == 1. These loci have become monomorphic for
  the mutant allele. Each generation, they are reset: p set to 0 (they
  become the new wild-type), and their phenotypic contribution is
  subtracted from `opt` so that `opt` is always interpreted relative to
  the current wild-type background. This re-centring avoids numeric drift
  of `opt` to large values.

---

### `fixed_loci_0` — Mask: Loci Without Mutant

- **Shape:** (L, rep) boolean
- **Meaning:** True where p == 0. Only these loci are eligible to receive
  a new forward mutation. Used to gate `mutation_mask`.

---

### `mutation_mask` — Mask: New Mutations This Generation

- **Shape:** (L, rep) boolean
- **Computed as:** `(np.random.rand(L, rep) < N*mu) & fixed_loci_0`
- **Meaning:** True for loci that receive a new mutation this generation.
  The probability N·μ per locus approximates a Poisson process (valid
  when N·μ ≪ 1). Where True: p is set to 1/N (one new copy in the
  population) and `effects` is re-drawn from N(0, a).

---

### `poly_loci` — Mask: Polymorphic Loci

- **Shape:** (L, rep) boolean
- **Computed as:** `~fixed_loci_0 & (p < 1 - 1/N)`
- **Meaning:** True for loci that are currently segregating (neither
  absent nor near-fixed). Only these undergo the back-mutation update,
  which can shift p by ±1/N with probabilities N·μ·(1−p) and N·μ·p
  respectively. New mutants (p = 1/N) are excluded because they were
  just introduced this step.

---

### `maxiter` — Number of Generations

- **Computed as:** `int(10 * N)`
- **Meaning:** Simulation run length. 10·N generations is chosen to be
  long enough for the population to reach mutation-selection-drift
  equilibrium (stationarity), since the characteristic time for fixation
  or loss of alleles is O(N) generations.

---

## 4. Output / Recorded Variables

---

### `Vg` — Additive Genetic Variance (final)

- **Shape:** (rep,) — one value per replicate
- **Formula (n-D):**

      Vg = (2 / n_traits) · Σ_l  ‖effects[l]‖²  ·  p[l]·(1−p[l])

- **Formula (1D):**

      Vg = 2·a² · Σ_l  p[l]·(1−p[l])

- **Meaning:** The additive genetic variance at the end of the simulation
  (after stationarity). This is the primary output. The factor 2 comes
  from diploid genetics (both allele copies contribute). Division by
  `n_traits` normalises to per-trait average V_g so that values are
  directly comparable across different dimensionalities.

---

### `hist` — Allele Frequency History

- **Shape:** (maxiter, L)
- **Meaning:** Records the allele frequency vector p at every generation
  for the single-replicate trajectory simulations. Used to plot
  trajectories of individual loci over time (Figure 1 panels).

---

### `Vg_hist` — Genetic Variance History

- **Shape:** (maxiter,)
- **Meaning:** Records V_g at every generation for trajectory simulations.
  Shows how genetic variance builds up from zero, fluctuates, and reaches
  an approximate steady state.

---

### `delta_norm` — Displacement Norm History (n-D only)

- **Shape:** (maxiter,)
- **Computed as:** `np.linalg.norm(opt − zbar)` each generation
- **Meaning:** The Euclidean distance between the current optimum and the
  population mean phenotype in T-dimensional trait space. Records how far
  the population lags behind the moving optimum. Plotted in Figure 1C.

---

### `delta` — Displacement History (1D only)

- **Shape:** (maxiter,)
- **Computed as:** `opt − zbar` (scalar)
- **Meaning:** 1D equivalent of `delta_norm`. The signed displacement of
  the mean phenotype from the optimum. Positive = population below
  optimum.

---

## 5. MPI Parallelisation Variables

---

### `comm` — MPI Communicator

- `MPI.COMM_WORLD`: the global communicator connecting all parallel processes.

### `size` — Number of MPI Processes

- Total cores allocated to the job. Each handles `rep_local` replicates.

### `rank` — This Process's ID

- Integer in [0, size−1]. Rank 0 is the **root** process: the only one
  that allocates `recvbuf`, calls `recvbuf.flatten()`, appends to
  `output`, and calls `np.savetxt`.

### `rep_local` — Replicates per Core

- `int(all_reps / size)`. Each core independently simulates this many
  replicate populations, then sends its `Vg_local` array to rank 0.

### `Vg_local` — Local Output Array

- Shape: (rep_local,). The V_g values computed by this core, sent to
  rank 0 via `comm.Gather`.

### `recvbuf` — Gather Receive Buffer

- Shape: (size, rep_local). Allocated only on rank 0. After
  `comm.Gather`, row r contains the `Vg_local` array from rank r.
  Flattened to (all_reps,) before appending to `output`.

### `output` — Collected Results

- Python list, one entry per parameter combination. Each entry is a
  1D array of length `all_reps` containing V_g from all replicate
  populations. Converted to a 2D array and saved with `np.savetxt`.

### `params` — Parameter Combination List

- Generated by `itertools.product(Ls, sigma_e2s, Ns, Vs, mus, a2s,
  thetas, n_traits, [rep_local])`.
- Each element is a tuple `(L, σ², N, V_s, μ, a², θ, T, rep)` passed
  to `simulate()`.

---

## 6. Parameter Sweep Arrays

These define the grid of conditions explored in the MPI scripts.

| Array | Values | Varied in |
|-------|--------|-----------|
| `sigma_e2s` | [0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2] | Both 1D and n-D |
| `Ls` | [100] | Both |
| `Ns` | [10 000] | Both |
| `Vs` | [5, 20] | Both |
| `mus` | [6.6e-6] | Both |
| `thetas` | [0.0] | Both |
| `a2s` | [0.1] | Both |
| `n_traits` (array) | [1, 2, 3, 5] | n-D only |

The outer product of all arrays gives the full set of parameter
combinations. The key axes of variation are **σ²** (x-axis in most
figures) and **T = n_traits** (separate curves/panels).

---

## 7. Key Equations Summary

**Allele frequency update (log-odds form):**
$$\ln\frac{p'}{1-p'} = \ln\frac{p}{1-p} + \frac{1}{V_s}\left(\mathbf{a}_l \cdot \boldsymbol{\delta}_t + \frac{1}{2}\|\mathbf{a}_l\|^2 (2p-1)\right)$$

**Wright-Fisher drift:**
$$p_{t+1} \sim \mathrm{Binomial}(N,\, p') / N$$

**Optimum movement (n-D, per component m):**
$$z^*_{m,t+1} = (1-\theta)\,z^*_{m,t} + \varepsilon_{m,t}, \quad \varepsilon_{m,t} \sim \mathcal{N}\!\left(0,\,\frac{\sigma^2}{T}\right)$$

**Additive genetic variance (per trait, n-D):**
$$V_g = \frac{2}{T}\sum_{l=1}^{L} \|\mathbf{a}_l\|^2\, p_l(1-p_l)$$

**Latter-Bulmer prediction (σ² = 0):**
$$V_g^{LB} = 4L\mu V_s$$

**Approximate prediction (σ² > 0):**
$$V_g \approx V_g^{LB} + \sqrt{V_s \sigma^2}$$
