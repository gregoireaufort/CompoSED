# Catalog Utilities

## Finite photometric grids

```{automodule} composed.catalog
:members: CatalogGridResult, PhotometricModelGrid, CatalogProfileGridResult, build_photometric_model_grid, evaluate_catalog_model_grid_likelihood, save_photometric_model_grid, load_photometric_model_grid, run_photometric_grid_catalog
:show-inheritance:
```

## Experimental rest-frame projection

```{automodule} composed.catalog_fast
:members: ExperimentalFastCatalogWarning, RestFrameSpectralGrid, RedshiftFilterOperator, NativeCatalogFitResult, build_restframe_spectral_grid, build_redshift_filter_operator, project_rest_grid_to_photometric_grid, fit_catalog_with_restframe_grid, save_restframe_spectral_grid, load_restframe_spectral_grid
:show-inheritance:
```
