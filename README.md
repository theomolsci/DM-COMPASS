**COMPASS** (COMmutativity Pre-screened Automated Selection of Scatterers) is a
dynamic, disentangled ansatz-construction protocol for the Variational Quantum
Eigensolver (VQE). It is built within the unitary coupled-cluster framework. The
goal is a compact, shallow-depth circuit that stays accurate for strongly
correlated molecules, where standard UCCSD fails.

**Key idea.** The ansatz is a product of two kinds of operators:
- **Cluster operators (τ)**: one- and two-body excitations that span the
  *primary* excitation manifold.
- **Scatterers (σ)**: two-body operators with effective hole–particle rank one.
  Each has a built-in projector, so it acts non-trivially only on selected
  primary-subspace determinants. Acting on a primary excitation, a scatterer
  can reach higher-rank (triple, quadruple, ...) excitations. These form the
  *secondary* manifold, obtained through nested commutators such as
  [σ, τ₂] → τ₃, without adding higher-rank operators to the circuit.

The ansatz has the form e^σ · e^τ |Φ_HF⟩. A scatterer contributes only if it
**does not commute** with a cluster operator that shares a contractible orbital
(CSO) with it. This commutativity criterion is used to prescreen scatterers.

**Workflow (see flowchart)**
1. **Select the best cluster operators.** Each double-excitation operator τ_I is
   evaluated with a one-parameter VQE, E_I = ⟨Φ_HF|e^(−τ_I) H e^(τ_I)|Φ_HF⟩.
   These evaluations are independent, so they can run in parallel on multiple
   quantum processors. Operators with |E_I − E_HF| > ε₁ are kept and sorted by
   energy lowering.
2. **Select scatterers via CSO scanning.** For each retained τ, candidate
   scatterers that share a CSO orbital with it are drawn from an operator bath.
   Each (τ, σ) pair is tested with a two-parameter VQE. The scatterer is kept
   only if it lowers the energy by more than a threshold ε₂.
3. **Assemble the ansatz.** The ansatz is built in energy order as
   [τ → its σ's] ..., followed by the single-excitation operators.
4. **Final VQE.** All parameters are optimized together with a classical
   optimizer (L-BFGS-B) to obtain the ground-state energy.

**Advantages:** much shallower circuits and fewer parameters than UCCSD or
generalized UCC. The screening steps are parallelizable. The method is resilient
to noise, which makes it suitable for NISQ devices.

## Citation

If you use this code or algorithm, please cite:

> D. Mondal, D. Halder, S. Halder, and R. Maitra, "Development of a compact
> Ansatz via operator commutativity screening: Digital quantum simulation of
> molecular systems," *J. Chem. Phys.* **159**, 014105 (2023).
> https://doi.org/10.1063/5.0153182

```bibtex
@article{mondal2023compass,
  author  = {Mondal, Dibyendu and Halder, Dipanjali and Halder, Sonaldeep and Maitra, Rahul},
  title   = {Development of a compact Ansatz via operator commutativity screening: Digital quantum simulation of molecular systems},
  journal = {The Journal of Chemical Physics},
  volume  = {159},
  number  = {1},
  pages   = {014105},
  year    = {2023},
  doi     = {10.1063/5.0153182}
}
```
