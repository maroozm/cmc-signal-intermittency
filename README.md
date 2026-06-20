## CMC Injection Workflow in This Directory

This document records the methodology, code choices, references, outputs, and caveats for the **CMC-style signal injection** work implemented in this directory.

---

### 1. Goal

The goal of this work is to construct a **hybrid background + critical-signal sample** inspired by the CMC/UrQMD methodology used in the intermittency literature, especially:

- `arXiv:2209.07135`  
  **Intermittency of charged particles in the hybrid UrQMD+CMC model at energies available at the BNL Relativistic Heavy Ion Collider**
- `arXiv:2412.06151`  
  **Identifying weak critical fluctuations of intermittency in heavy-ion collisions with topological machine learning**

In those works, a weak critical signal generated from a **Critical Monte Carlo (CMC)** model is embedded into a realistic heavy-ion background by replacing a small fraction of particles in each event.

In this directory, that idea is adapted to an **EPOS event tree**.

---

### 2. Conceptual background

#### 2.1 What is the CMC signal?

The CMC model is used to generate **self-similar, clustered critical fluctuations**. This is done through a **Lévy random walk** whose step-size distribution follows

\[
\rho(r) \propto r^{-1-\mu}
\]

with parameters corresponding to a critical system in the 3D Ising universality class.

In the charged-particle intermittency context:

- \(\mu = 1/6\)
- \(r_{\min}/r_{\max} = 10^{-7}\)

The generated point cloud is intended to carry critical-like scale-invariant structure.

---

#### 2.2 In what space is the signal generated?

A key methodological point — and a deliberate departure from the published papers:

- the published hybrid UrQMD+CMC method generates the signal in **transverse momentum space** \((p_x, p_y)\)
- **this implementation generates the signal in pseudorapidity–azimuth space** \((\eta, \phi)\)

This choice was made because the scaled factorial moments (SFMs) in this workflow are computed in \((\eta, \phi)\) — so the CMC clustering and the analysis live in the **same coordinate space**. The Lévy walk generates a self-similar point pattern regardless of the 2D coordinate system used; what matters is that the walk and the SFM analysis are done in matching spaces.

#### Walk geometry in \((\eta, \phi)\) space

- **Seed**: \((\eta_0, \phi_0)\) from a randomly chosen accepted track in the event
- Each step: draw a step size \(r\) from \(\rho(r)\), draw a random direction \(\theta \in [0, 2\pi)\), update:
  \[
  \eta_1 = \eta_0 + r\cos\theta, \quad \phi_1 = \phi_0 + r\sin\theta
  \]
- \(\phi\) is **wrapped** to \([0, 2\pi)\) after each step — this handles the circular periodicity of the azimuthal angle
- Steps that would place \(|\eta| \geq \eta_{\max}\) are **rejected** (walk stays inside the acceptance)

##### Step range parameters

\[
r_{\max} = 2 \times \eta_{\max} = 1.0
\]
\[
r_{\min} = r_{\max} \times 10^{-7}
\]

The maximum step covers the full \(\eta\) window in one step, and the ratio \(r_{\min}/r_{\max} = 10^{-7}\) matches the papers' convention.

---

#### 2.3 What does the "2%" or "1.7%" mean in the papers?

The papers define a replacement ratio

\[
\lambda = \frac{N_{\rm CMC}}{N_{\rm UrQMD}}
\]

This means:

- **it is a particle-level replacement fraction within each event**
- **it is not a fraction of events**

So:

- `λ = 2%` means about 2% of the accepted particles in an event are replaced by CMC particles
- `λ = 5%` means about 5% are replaced
- `λ ≈ 1–2%` is the weak-signal range used in the earlier hybrid UrQMD+CMC physics interpretation

In this directory, the currently implemented value is:

- `_lam = 0.02`  
  i.e. **2%**

---

### 5. Current implementation in `ep.cmc.injection.C`

### 5.1 Event reading

The macro:

1. reads the input file path from `files.txt`
2. opens the ROOT file
3. reads the tree `teposevent0`
4. binds the branch buffers
5. allocates branch arrays sized from `tree->GetMaximum("np")`

This dynamic sizing was introduced to avoid an earlier segmentation fault caused by fixed-size branch buffers.

---

#### 5.2 Accepted-track selection

Tracks are accepted if they satisfy:

- charged particle:
  - `EPOSParticleInfo::is_charged(pid)`
- status code:
  - `ist == 8`
- transverse momentum:
  - `0.2 < pT < 3.0`
- pseudorapidity:
  - `|eta| < 0.5`

These are encoded as:

- `_pt_min = 0.2`
- `_pt_max = 3.0`
- `_eta_max = 0.5`

The helper functions are:

- `getPt(px, py)`
- `getEta(pt, pz)`
- `getPhi(px, py)` — returns \(\phi \in [0, 2\pi)\)

---

#### 5.3 CMC candidate generation

The CMC pool is generated using:

```cpp
makePool(seedEta, seedPhi, requested, rng, hLevyStep)
```

The procedure is:

1. choose a seed point \((\eta_0, \phi_0)\) from a randomly selected accepted track
2. include the seed itself as the first pool entry
3. repeatedly draw a Lévy step size from
   \[
   \rho(r) \propto r^{-1-\mu}
   \]
4. draw a random direction \(\theta \in [0,2\pi)\)
5. update:
   \[
   \eta \to \eta + r\cos\theta, \quad \phi \to \phi + r\sin\theta
   \]
6. reject steps outside the \(\eta\) acceptance, wrap \(\phi\) to \([0, 2\pi)\)
7. store accepted candidates as `CMCCand{eta, phi}`

Current parameter values:

- `_mu = 1.0 / 6.0`
- `_ep_rmax = 2.0 * _eta_max = 1.0`
- `_ep_rmin = _ep_rmax * 1.0e-7`

---

#### 5.4 Number of replacements per event

The number of requested replacements is sampled from a binomial distribution:

```cpp
rng.Binomial(acceptedCount, _lam)
```

This means the actual per-event replacement fraction fluctuates around the target \(\lambda\), as expected statistically.

---

#### 5.5 Injection / matching logic

For each event:

1. gather all accepted tracks
2. choose one accepted track randomly as the CMC seed — use its \((\eta, \phi)\) as the walk starting point
3. shuffle the accepted tracks
4. truncate the shuffled list to the requested number of replacements (targets)
5. generate a CMC candidate pool (oversized for robustness)
6. **sequentially assign** pool candidates to targets: `target[i]` gets `pool[i]`

There is **no pT-tolerance matching** — because pT is preserved exactly by construction. The injection reconstructs full 3-momentum from the CMC candidate's angular position and the target track's original pT:

\[
p_x^{\rm new} = p_T^{\rm orig} \cos(\phi_{\rm CMC})
\]
\[
p_y^{\rm new} = p_T^{\rm orig} \sin(\phi_{\rm CMC})
\]
\[
p_z^{\rm new} = p_T^{\rm orig} \sinh(\eta_{\rm CMC})
\]

---

#### 5.6 What is actually replaced?

In the current implementation:

- `px` is replaced
- `py` is replaced
- `pz` is replaced

while the following are **kept unchanged**:

- `id`
- `ist`

##### Interpretation

This means the code injects a **critical-like angular structure** while preserving the transverse momentum magnitude of each replaced track. The particle's position in \((\eta, \phi)\) space is moved to the CMC-assigned location, while its \(p_T\) and particle identity remain those of the original EPOS track.

The `pz` write-back is essential — without it, the \(\eta\) of the injected particle would not change and the clustering would remain invisible in the \((\eta, \phi)\) analysis.

##### Caveat

This is **not** a full dynamical re-generation of particles. It is a controlled angular-structure embedding procedure that preserves the pT spectrum exactly.

---

### 6. Current outputs

The macro writes an injected tree output and a QA ROOT file.

#### 6.1 Injected tree output

The output filename is derived from the input filename by appending:

- `.cmc_lambda<ratio>.root`

For the current `λ = 0.02`, an input file

- `something.root`

becomes

- `something.cmc_lambda0p02.root`

Important:

- because the input file path from `files.txt` is absolute, the injected output tree file is written alongside that input path, not necessarily into `./`

#### 6.2 QA ROOT output

The macro also writes a QA ROOT output named with the pattern:

- `<basename>.cmc_lambda0p02.qa.root`

This file is created in the working directory and contains:

- track-count histograms (accepted, requested, injected per event)
- pT, η, φ distributions before and after injection (both for all accepted tracks and for replaced tracks specifically)
- Δpx, Δpy, ΔR displacement histograms
- Lévy step size distribution
- pool size and unmatched target histograms
- 2D (px,py) and (η,φ) scatter histograms before/after
- scaled factorial moments F₂ through F₅ computed in (η,φ) space, both before and after injection
- THnSparse with per-track (η, φ, bim, event) for detailed downstream analysis

---

### 7. Scaled factorial moment computation

The SFMs are computed in **2D \((\eta, \phi)\) space**.

The analysis window is partitioned into \(M \times M\) cells:

- \(\eta \in [-\eta_{\max}, +\eta_{\max}]\)
- \(\phi \in [0, 2\pi)\)

with \(M = 2(k+2)\) for \(k = 0, \ldots, 51\) (i.e. \(M = 4, 6, 8, \ldots, 106\)).

The \(q\)-th order SFM is:

\[
F_q(M) = \frac{\langle \frac{1}{M^2} \sum_i n_i(n_i-1)\cdots(n_i-q+1) \rangle}{\langle \frac{1}{M^2} \sum_i n_i \rangle^q}
\]

where \(n_i\) is the multiplicity in the \(i\)-th cell and the average is over events.

**Implementation note**: The falling factorial \(n(n-1)\cdots(n-q+1)\) is computed iteratively (not via `TMath::Factorial`) to avoid double-precision overflow at high bin occupancy. An earlier version used `TMath::Factorial(n)/TMath::Factorial(n-q)`, which overflowed and silently discarded events carrying the strongest CMC signal.

---

### 9. References consulted for this implementation

#### Primary references

##### 1. Rui Wang et al.

**Identifying weak critical fluctuations of intermittency in heavy-ion collisions with topological machine learning**  
`arXiv:2412.06151`  
Phys. Lett. B 864, 139405 (2025)

Why it matters here:

- gives the recent signal/background construction language
- clearly describes the replacement-ratio workflow
- uses 5% and 10% signal events for topological ML studies

---

##### 2. Jin Wu et al.

**Intermittency of charged particles in the hybrid UrQMD+CMC model at energies available at the BNL Relativistic Heavy Ion Collider**  
`arXiv:2209.07135`  
Phys. Rev. C 106, 054905 (2022)

Why it matters here:

- this is the direct hybrid UrQMD+CMC construction paper
- defines the replacement fraction \(\lambda\)
- motivates the weak-signal embedding picture
- supports the 1–2% physics-style setting

---

##### 3. Jin Wu et al.

**Probing QCD critical fluctuations from intermittency analysis in relativistic heavy-ion collisions**  
`arXiv:1901.11193`  
Phys. Lett. B 801 (2020) 135186

Why it matters here:

- documents the CMC / Lévy-random-walk critical-fluctuation picture
- gives context for the 3D Ising / intermittency interpretation

---

### 10. Current limitations / caveats

#### 10.1 `px`, `py`, `pz` are replaced; `id`, `ist` are kept

The injected tracks get new angular coordinates but keep their original particle identity and pT magnitude.

#### 10.2 The signal is generated in \((\eta, \phi)\), not \((p_x, p_y)\) as in the published papers

This is a deliberate methodological choice. The published UrQMD+CMC papers generate the Lévy walk in transverse momentum space and compute SFMs there. This implementation generates and analyses in \((\eta, \phi)\) instead. The Lévy walk's self-similar properties are coordinate-independent, but the intermittency indices and scaling may differ quantitatively from the published values.

#### 10.3 The method is a practical implementation, not the authors' exact private production code

It is based on the published methodology and adapted to the available EPOS tree structure.

#### 10.4 No mixed-event subtraction is performed in the macro

The macro computes raw \(F_q(M)\) both before and after injection. The correlator \(\Delta F_q(M) = F_q^{\rm data} - F_q^{\rm mixed}\) used in the published analyses is not computed here — that subtraction must be performed downstream.

---

### 11. Suggested publication narrative

If this work is written up, a reasonable methodology section could say:

1. EPOS events were used as the baseline background sample.
2. Charged particles were selected within:
   - `|eta| < 0.5`
   - `0.2 < pT < 3.0 GeV/c`
3. A weak critical-like signal was generated in pseudorapidity–azimuth space \((\eta, \phi)\) using a Lévy random walk with:
   - `mu = 1/6`
   - `r_min/r_max = 10^-7`
4. For each event, a number of accepted particles sampled from a binomial distribution with target replacement ratio `λ` were chosen for replacement.
5. The momentum of each replaced track was reconstructed from the CMC-assigned \((\eta, \phi)\) and the track's original \(p_T\), preserving the transverse momentum spectrum exactly.
6. All three momentum components `px`, `py`, `pz` were updated; `id` and `ist` were left unchanged.

---