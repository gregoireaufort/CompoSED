# Backends

## Common contract

```{automodule} composed.backends.base
:members: ModelPhotometry, ModelSpectrum, SEDBackend
:show-inheritance:
```

## FSPS

```{automodule} composed.backends.fsps
:members: FSPSBackend
:show-inheritance:
```

## CIGALE

```{automodule} composed.backends.cigale
:members: CIGALEBackend, build_cigale_parameter_space, build_cigale_backend_and_parameter_space
:show-inheritance:
```

## Deterministic mock

```{automodule} composed.backends.mock
:members: MockBackend
:show-inheritance:
```
