# Model Variables Reference

## 1D Model (supervisor's paper)

| Symbol | Code name | Formula / Definition | Meaning |
|--------|-----------|----------------------|---------|
| z | — | — | Phenotypic trait value of a diploid individual |
| z* | `opt` | z*_{t+1} = (1−θ)·z*_t + ε_t | Trait optimum — value of z with highest fitness |
| z̄ | `zbar` | z̄ = Σ_l a·sign_l·(2p_l² + p_l(1−p_l)) | Population mean phenotype |
| δ_t | `opt - zbar` | δ_t = z* − z̄ | Displacement of population mean from optimum |
| W | — | W ∝ exp(−(z−z*)² / 2V_s) | Individual fitness under stabilizing selection |
| V_s | `V_s` | — | Selection strength. Larger V_s = **weaker** selection |
| σ² | `sigma_e2` | σ² = Var(ε_t), ε_t ~ N(0, σ²) | Fluctuation intensity = variance of optimum shift per generation |
| E[δ²] | — | E[δ²] ≈ V_s · σ² / V_g | Expected squared displacement at steady state |
| L | `L` | — | Number of loci (genes) contributing to the trait |
| N | `N` | — | Population size (haploid count; paper uses 2N for diploid) |
| μ | `mu` | — | Mutation rate per locus per generation |
| a² | `a2` | — | Effect size variance. Each allele changes z by ±a |
| a | `a` | a = √a2 | Effect size magnitude |
| θ | `theta` | — | OU restoring force on optimum. θ=0 → pure Brownian motion |
| p_l | `p` | — | Frequency of mutant allele at locus l (0=absent, 1=fixed) |
| sign_l | `sign` | ∈ {−1, +1} | Direction of allele effect (positive or negative on trait) |
| V_g | `Vg` | V_g = 2a² · Σ_l p_l(1−p_l) | Additive genetic variance |
| V_g^LB | — | V_g^LB = 4LμV_s | Latter-Bulmer prediction (no fluctuation, σ²=0) |
| h² | — | h² = V_g / (1 + V_g) | Narrow-sense heritability (environmental variance = 1) |

---

## n-D Extension (your work)

| Symbol | Code name | Formula / Definition | Meaning |
|--------|-----------|----------------------|---------|
| T | `n_traits` | — | Number of trait dimensions under simultaneous selection |
| **z** | — | vector of length T | Phenotypic vector of an individual |
| **z*** | `opt` | shape (T, rep) | Optimum vector in T-dimensional trait space |
| **a**_l | `effects[l]` | ~ N(0, a) per component | Effect vector of locus l — one component per trait |
| \|\|**a**_l\|\|² | `norm2` | Σ_t effects[l,t]² ≈ T·a² | Squared magnitude of locus l's effect across all traits |
| **δ**_t | `opt - zbar` | **δ**_t = **z*** − **z̄** | T-dimensional displacement vector |
| \|\|**δ**_t\|\| | `delta_norm` | √(Σ_t δ_t²) | Norm of displacement (plotted in Figure 1 Panel C) |
| σ²/T | `sigma_e2/n_traits` | Var per component = σ²/T | Per-component optimum fluctuation (normalised so total = σ²) |
| E[\|\|**δ**\|\|²] | — | ≈ T · V_s · σ² / V_g | Expected squared displacement magnitude |
| V_g (per trait) | `Vg` | (2/T) · Σ_l \|\|**a**_l\|\|² · p_l(1−p_l) | Per-trait average genetic variance (comparable to 1D) |

---

## Key Equations

**Optimum movement (1D):**
$$z^*_{t+1} = (1-\theta) z^*_t + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0,\, \sigma^2)$$

**Optimum movement (n-D, per component):**
$$z^*_{m,t+1} = (1-\theta) z^*_{m,t} + \varepsilon_{m,t}, \quad \varepsilon_{m,t} \sim \mathcal{N}\!\left(0,\, \frac{\sigma^2}{T}\right)$$

**Latter-Bulmer genetic variance (σ²=0):**
$$V_g^{LB} = 4L\mu V_s$$

**Approximate genetic variance (σ²>0):**
$$V_g \approx V_g^{LB} + \sqrt{V_s \sigma^2}$$

**Heritability:**
$$h^2 = \frac{V_g}{1 + V_g}$$

**Allele frequency change (stabilizing + directional selection):**
$$\Delta \ln\!\frac{p}{1-p} = \frac{1}{V_s}\left(\mathbf{a}_l \cdot \boldsymbol{\delta}_t + \frac{1}{2}\|\mathbf{a}_l\|^2 (2p-1)\right)$$
