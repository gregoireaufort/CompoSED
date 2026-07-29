# Catalog Workflows

There are two distinct catalog strategies.

## Amortized neural posterior

Train a MAF or MDN once and batch many observations:

```python
samples = posterior.sample(
    catalog_flux,
    sigma=catalog_sigma,
    input_units="native",
    num_samples=128,
    batch_size=8192,
    seed=5,
)
```

This is the intended high-throughput path for large catalogs. The trained
posterior is valid only over the simulated prior, noise model, bands, units,
and selection regime represented during training.

## Reusable finite model grids

For finite CIGALE-style parameter supports:

```python
catalog_result = run_photometric_grid_catalog(
    backend,
    datasets,
    parameter_space,
    filters=filters,
    model_chunk_size=2048,
    object_chunk_size=512,
)
```

The backend is evaluated once per grid point. Likelihoods are then broadcast
over objects in chunks. All objects must share band order and flux unit, but
may have different masks and upper-limit states.

The grid path supports only finite-valued parameters. It does not silently
convert a continuous prior into an arbitrary grid.

## Experimental fast projection

`composed.catalog_fast` tabulates rest-frame spectra and projects them through
redshift/filter operators. It is restricted to backends whose rest SED is
independent of redshift and requires complete wavelength coverage. It is not a
drop-in replacement for normal backend evaluation.

## Catalog audit checklist

- Is object selection performed before or after adding noise?
- Are bands, units, masks, and upper-limit conventions identical across rows?
- Does the training/simulation distribution cover the selected catalog?
- Are model-grid parameters finite choices or continuous quadrature axes?
- Is the mass prior applied as a prior rather than induced by grid spacing?
- Are failed objects reported rather than silently dropped?
