# Data, Masks, And Filters

## Photometry

`SEDDataset` holds one object's photometry in linear flux units:

```python
data = SEDDataset(
    band_names=["u", "g", "r", "i"],
    flux=np.array([np.nan, 0.12, 0.25, 0.31]),
    sigma=np.array([0.03, 0.02, 0.02, 0.03]),
    mask=np.array([True, True, True, False]),
    upper_limit=np.array([0.06, 0.0, 0.0, 0.0]),
    upper_limit_mask=np.array([True, False, False, False]),
    flux_unit="maggies",
)
```

The arrays have shape `(n_band,)`. A band is active only when:

- the explicit mask permits it;
- `sigma` is finite and strictly positive;
- a detection has finite `flux`, or an upper limit has finite `upper_limit`.

At least one active band is required. An upper-limit band may use `NaN` as the
detection flux because that value does not enter the residual term.

`True` means **included** in `mask`. In `upper_limit_mask`, `True` means the band
is treated as a censored upper limit rather than a detection.

## Upper limits and AB depths

The likelihood consumes an upper flux limit $L$, not a magnitude. An AB limiting
magnitude is a faintness threshold and maps to a maximum allowed linear flux:

```python
limit_maggies = 10.0 ** (-0.4 * limiting_ab_magnitude)
```

The associated `sigma` sets the width of the Gaussian CDF likelihood. Document
whether the quoted survey depth is 1-sigma, 3-sigma, 5-sigma, or another
convention before converting it.

## Spectra

`SpectrumDataset` uses:

- observed wavelength in Angstrom;
- observed $f_\lambda$ in
  `erg s^-1 cm^-2 Angstrom^-1`;
- one-dimensional arrays of equal shape;
- a strictly increasing wavelength grid.

```python
spectrum = SpectrumDataset(
    wavelength=wave_angstrom,
    flux=flam_cgs,
    sigma=flam_sigma_cgs,
    mask=good_pixel,
)
```

The mask is applied to wavelength, flux, and sigma together.

## Joint data

```python
data = SpectroPhotometricDataset(
    photometry=photometry,
    spectrum=spectrum,
)
```

The joint likelihood adds the photometric and spectral terms and adds the prior
once.

## Filters

`FilterSet` fixes filter objects and their names in one order. For sedpy:

```python
filters = load_filter_set(["sdss_u0", "sdss_g0", "sdss_r0"])
```

CIGALE can also consume its native database filter names through a `FilterSet`.
Model photometry is aligned to data by unique band name; duplicate names are
rejected instead of being counted twice.

Filter wavelengths, transmission convention, and source database are part of
the scientific model and should be included in provenance.
