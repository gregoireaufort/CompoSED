# Likelihoods

## Gaussian detections

For active detections, CompoSED evaluates

$$
\log p(d \mid \theta)
= -\frac{1}{2}\sum_i
\left[
\frac{(d_i-m_i(\theta))^2}{\sigma_{\mathrm{eff},i}^2}
+ \log\left(2\pi\sigma_{\mathrm{eff},i}^2\right)
\right].
$$

The effective photometric uncertainty is

$$
\sigma_{\mathrm{eff},i}^2
= \sigma_{\mathrm{catalog},i}^2
+ \sigma_{\mathrm{floor}}^2
+ \left(\eta\,m_i(\theta)\right)^2,
$$

where `eta` is `photometric_model_discrepancy`.

```python
likelihood = Gaussian(
    photometric_sigma_floor=1.0e-12,
    photometric_model_discrepancy=0.03,
)
```

The catalog sigma is not modified in place. Because the discrepancy term
depends on the model, its log-determinant contribution is retained.

## Censored upper limits

For an upper flux limit $L_i$, the likelihood contribution is

$$
\log \Phi\left(\frac{L_i-m_i(\theta)}
{\sigma_{\mathrm{eff},i}}\right),
$$

where $\Phi$ is the standard normal CDF. It is not a Gaussian residual at a
fabricated flux value.

## Prior and posterior

Likelihood objects expose the conceptual pieces separately:

```python
problem.log_prior(theta)
problem.log_likelihood(theta)
problem.log_posterior(theta)
```

`log_prob` is a compatibility spelling for the posterior. Samplers that own
the prior internally, such as PocoMC, receive `log_likelihood` and the
translated `ParameterSpace` prior so the prior is applied exactly once.

## Mass normalization

If the backend declares `PER_SOLAR_MASS`, CompoSED requires `log10_mass` and
uses

$$
m_i^\mathrm{absolute}
= 10^{\mathrm{log10\_mass}} m_i^\mathrm{per\ mass}.
$$

If the backend declares `ABSOLUTE`, no mass factor is applied, even when a
parameter named `log10_mass` is present. Missing mass for a per-mass backend is
an error; implicit inference from class names or flux scale is forbidden.

## Spectra and joint fits

```{admonition} Experimental interface
:class: warning
Only the photometric likelihood is release-ready. Spectral and joint
spectrophotometric likelihoods are retained for development and validation.
```

The spectral likelihood uses active observed pixels on the supplied wavelength
grid and the same mass-normalization rule. A joint
`SpectroPhotometricDataset` sums the two data likelihoods and adds the prior
once.

The current first-pass spectral likelihood is diagonal Gaussian. Covariance
matrices, calibration nuisance models, and instrumental line-spread
convolution are not implemented, which is why this path is experimental.
