# Validation Provenance Checklist

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
        "cue_data_dir": cue_data_dir,
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

Plotting or downstream validation cells should fail loudly when the sidecar is
missing:

```python
from composed.provenance import require_provenance

require_provenance(output_dir / "reference_spectra.npz")
data = np.load(output_dir / "reference_spectra.npz")
```

The sidecar records:

- git commit, branch, dirty flag, and porcelain status;
- Python executable and version;
- versions of the main numerical/SPS/inference packages when available;
- selected environment variables such as `SPS_HOME`, `DSPS_CONTINUUM_SSP_FILE`,
  and `CUE_DATA_DIR`;
- SHA256 hashes of declared files or directories;
- random seed;
- command arguments;
- any extra stage-specific metadata.

For science validation, treat SSP grids, Cue data directories, filter curves,
catalogs, and cached reference spectra as model inputs. They should be included
in `provenance_paths` whenever they affect the plotted result.
