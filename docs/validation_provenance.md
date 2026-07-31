# Validation Provenance Checklist

```{admonition} Spectral artifacts
:class: note
References to spectra on this page concern experimental backend-validation
artifacts. They do not make the spectral likelihood a stable fitting interface;
only the pure photometric pipeline is release-ready.
```

Validation plots are only useful if the cached arrays behind them can be traced
back to the code and model inputs that produced them.

For notebook-level or script-level validation products, save a provenance
sidecar next to each generated `.npz` product:

```python
from composed.provenance import save_npz_with_provenance

save_npz_with_provenance(
    output_dir / "reference_spectra.npz",
    provenance_paths={
        "ssp_file": ssp_file,
        "input_catalog": catalog_path,
    },
    seed=seed,
    command_args=vars(args),
    extra={"stage": "references"},
    rest_wave_nm=rest_wave_nm,
    spectrum_model=spectrum_model,
)
```

This writes:

- `reference_spectra.npz`: numerical arrays;
- `reference_spectra.provenance.json`: code, environment, and input hashes.

The sidecar also stores the SHA256 of `reference_spectra.npz` itself. The
default loader therefore detects a cache that was edited, truncated, or
replaced after the provenance was written.

Plotting or downstream validation cells should fail loudly when the sidecar is
missing:

```python
from composed.provenance import require_provenance

require_provenance(output_dir / "reference_spectra.npz")
data = np.load(output_dir / "reference_spectra.npz")
```

`load_inference_result`, `load_photometric_model_grid`, and
`load_restframe_spectral_grid` perform the equivalent verification by default.
Their explicit `verify_provenance=False` or
`require_provenance_sidecar=False` escape hatches exist only for inspecting
legacy products; do not use an unverified artifact for a reported result.

The sidecar records:

- git commit, branch, dirty flag, and porcelain status;
- Python executable and version;
- versions of the main numerical/SPS/inference packages when available;
- selected environment variables such as `SPS_HOME`;
- SHA256 hashes of declared files or directories;
- random seed;
- command arguments;
- any extra stage-specific metadata.

`Problem.specification()` separately fingerprints the scientific calculation:
data/noise arrays, units, masks, priors, backend configuration and engine
versions, filter curves, and parameter-transform code including referenced
global values. It uses engine source hashes and revision information rather
than absolute installation paths, so equivalent calculations can be compared
across machines. The sidecar still records local paths for provenance and
debugging.

For science validation, treat SSP grids, filter curves, catalogs, and cached
reference spectra as model inputs. They should be included in
`provenance_paths` whenever they affect the plotted result.
