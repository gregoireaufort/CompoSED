from __future__ import annotations

import os


def _env_flag(name: str) -> bool | None:
    value = os.environ.get(name)
    if value is None or value.strip().lower() in {"", "auto"}:
        return None
    if value.strip().lower() in {"1", "true", "yes", "on", "float64", "x64"}:
        return True
    if value.strip().lower() in {"0", "false", "no", "off", "float32"}:
        return False
    raise ValueError(f"{name} must be one of auto/true/false, got {value!r}.")


def require_jax():
    """Import JAX lazily with a targeted error message."""

    try:
        requested_x64 = _env_flag("COMPOSED_JAX_ENABLE_X64")
        requested_platform = os.environ.get("JAX_PLATFORM_NAME", "").strip()
        if not requested_platform:
            requested_platform = os.environ.get("JAX_PLATFORMS", "").split(",")[0].strip()
        requested_platform = requested_platform.lower()
        if requested_x64 is not None:
            os.environ["JAX_ENABLE_X64"] = "True" if requested_x64 else "False"
        elif requested_platform in {"mps", "metal"}:
            os.environ["JAX_ENABLE_X64"] = "False"

        import jax

        # Default to float64 on CPU/CUDA for validation, but do not request
        # float64 on Apple's Metal/MPS backend. MPS does not support float64,
        # and trying to move float64 arrays there fails before any science can
        # happen. Users can override this with COMPOSED_JAX_ENABLE_X64.
        enable_x64 = requested_x64
        if enable_x64 is None:
            platform = os.environ.get("JAX_PLATFORM_NAME")
            if platform is None:
                platform = jax.default_backend()
            enable_x64 = platform.lower() not in {"mps", "metal"}
        jax.config.update("jax_enable_x64", bool(enable_x64))
        import jax.numpy as jnp
    except ImportError as exc:
        raise ImportError(
            "composed.experimental.jaxcigale requires optional JAX dependencies. "
            "Install them with something like `pip install jax jaxlib`, or use "
            "`pip install composed[jaxcigale]` once the optional extra is available."
        ) from exc
    return jax, jnp


def require_dsps():
    """Import DSPS lazily for the DSPS stellar module."""

    try:
        import dsps
    except ImportError as exc:
        raise ImportError(
            "DSPSStellarModule requires the optional `dsps` package and SSP data. "
            "Install dsps and provide DSPS SSP templates before using this module."
        ) from exc
    return dsps


def require_numpyro():
    """Import NumPyro lazily for the first NUTS runner."""

    try:
        import numpyro
        from numpyro.infer import MCMC, NUTS
    except ImportError as exc:
        raise ImportError(
            "run_numpyro_nuts requires optional `numpyro`. Install numpyro to run "
            "JAX-CIGALE NUTS demos."
        ) from exc
    return numpyro, MCMC, NUTS
