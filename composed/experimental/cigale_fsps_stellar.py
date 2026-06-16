from __future__ import annotations

from pathlib import Path


MIXED_GRID_NOCACHE_MODULES = (
    "fsps_stellar",
    "nebular",
    "dustatt_modified_starburst",
    "redshifting",
)
"""CIGALE modules to keep uncached when comparing BC03 and FSPS grids."""


def module_directory() -> Path:
    """Return the directory containing experimental CIGALE module files."""

    return Path(__file__).with_name("cigale_modules")


def register_cigale_fsps_stellar_module() -> Path:
    """Make the experimental ``fsps_stellar`` module visible to CIGALE.

    The helper does not copy files into a CIGALE checkout. Instead it appends
    ``composed``'s experimental module directory to ``pcigale.sed_modules``'
    import path, so CIGALE can resolve ``fsps_stellar`` as if it lived beside
    its built-in modules.

    Returns
    -------
    pathlib.Path
        The path that was registered.
    """

    try:
        import pcigale.sed_modules as sed_modules
    except ImportError as exc:
        raise ImportError("register_cigale_fsps_stellar_module requires pcigale.") from exc

    path = module_directory()
    path_string = str(path)
    if path_string not in sed_modules.__path__:
        sed_modules.__path__.append(path_string)
    return path


def mixed_grid_nocache_modules() -> list[str]:
    """Return CIGALE modules that should be uncached for mixed-grid diagnostics.

    CIGALE modules can cache wavelength-grid-dependent arrays on module
    instances.  That is normally a performance feature, but it is hazardous
    when a single script alternates between BC03 and FSPS stellar wavelength
    grids.  Use this list for BC03-vs-FSPS comparison scripts.
    """

    return list(MIXED_GRID_NOCACHE_MODULES)


def make_mixed_grid_sed_warehouse():
    """Return a ``SedWarehouse`` configured for BC03/FSPS comparison scripts."""

    register_cigale_fsps_stellar_module()
    try:
        from pcigale.warehouse import SedWarehouse
    except ImportError as exc:
        raise ImportError("make_mixed_grid_sed_warehouse requires pcigale.") from exc
    return SedWarehouse(nocache=mixed_grid_nocache_modules())


__all__ = [
    "MIXED_GRID_NOCACHE_MODULES",
    "make_mixed_grid_sed_warehouse",
    "mixed_grid_nocache_modules",
    "module_directory",
    "register_cigale_fsps_stellar_module",
]
