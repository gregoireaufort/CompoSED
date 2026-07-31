# Scientific Conventions

This page is the compact convention sheet for auditing a CompoSED analysis.

## Photometry

- Data and model photometry are linear flux densities.
- The canonical backend/likelihood unit is **maggies**.
- One maggie is the AB zero-point flux density; $m_\mathrm{AB}=-2.5\log_{10}f$
  when $f$ is in maggies.
- Band names are unique and are used for model/data alignment.
- Masks use `True = included`.
- Upper limits are flux ceilings evaluated with a Gaussian CDF.

## Spectra

These conventions describe the experimental spectral interface. Only the
photometric pipeline is release-ready.

- Wavelengths are observed-frame Angstrom.
- Flux is observed $f_\lambda$ in
  `erg s^-1 cm^-2 Angstrom^-1`.
- Wavelength arrays are one-dimensional and strictly increasing.
- The current first-pass spectral Gaussian likelihood assumes diagonal
  uncertainties and does not yet model covariance, calibration nuisance terms,
  or a complete instrumental line-spread function.

## Redshift

FSPS and CIGALE backends accept `z`, `zred`, or `redshift`, with a declared
default key. The likelihood does not interpret redshift names. Use one name
consistently in a `ParameterSpace`.

## Stellar mass

The stable meaning of `log10_mass` is:

$$
\log_{10}\left(M_\star^\mathrm{surviving}/M_\odot\right).
$$

For `PER_SOLAR_MASS` backends, model output is per one solar mass of surviving
stellar mass and the likelihood applies `10**log10_mass` once. For `ABSOLUTE`
backends, the likelihood applies no mass factor.

Formed mass may appear in backend metadata, but it is not the public fitted
mass parameter.

## SFH

Named SFHs produce:

- a strictly increasing time-since-onset grid in Gyr;
- finite non-negative SFR in solar masses per year;
- explicit age and normalization metadata.

For per-mass backends, histories are normalized internally and converted to the
surviving-mass output convention. See {doc}`../sfh_models`.

## Cosmology

FSPS defaults to Astropy Planck18 for luminosity distance and cosmic age.
CIGALE v2022.0's native redshifting module uses WMAP7. A custom CIGALE backend
cosmology can affect named-SFH age conversion but cannot replace the upstream
redshifting cosmology.

This difference must be recorded when comparing engines.

## Failure semantics

- Physically invalid parameter states raise `ModelDomainError` in backends or
  transforms.
- Public posterior evaluation converts controlled domain failures to `-inf`.
- Shape, unit, missing-parameter, and configuration errors raise explicitly.
- Simulators raise controlled errors for invalid model outputs rather than
  emitting NaN training rows.

## Reproducibility

Random seeds, active masks, catalog cuts, filters, engine versions, external
model-grid locations/hashes, and result fingerprints are part of the scientific
record. A visually plausible plot is not a substitute for checking these
inputs.
