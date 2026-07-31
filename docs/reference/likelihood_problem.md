# Likelihoods And Problems

```{admonition} Stability
:class: warning
`GaussianPhotometricLikelihood` and photometric `Problem` usage are stable.
`GaussianSpectralLikelihood`, spectral simulation, and joint
spectrophotometric problems are experimental.
```

```{automodule} composed.likelihood
:members: GaussianPhotometricLikelihood, GaussianSpectralLikelihood, PhotometricSimulationError, SpectralSimulationError, effective_photometric_sigma
:show-inheritance:
```

```{automodule} composed.problem
:members: Gaussian, Problem
:show-inheritance:
```

```{automodule} composed.errors
:members: ModelDomainError
:show-inheritance:
```
