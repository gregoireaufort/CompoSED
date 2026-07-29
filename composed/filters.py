from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class FilterSet:
    """Ordered collection of backend filter objects and unique band names.

    Parameters
    ----------
    filters
        Backend-native filter objects or names.
    names
        Optional explicit band names. When omitted, names are read from each
        object's ``name`` attribute where available.
    """

    filters: Sequence[object]
    names: Sequence[str] | None = None

    def __post_init__(self) -> None:
        filters = tuple(self.filters)
        if self.names is None:
            names = tuple(f if isinstance(f, str) else getattr(f, "name", str(i)) for i, f in enumerate(filters))
        else:
            names = tuple(str(name) for name in self.names)
        if len(names) != len(filters):
            raise ValueError("names length must match filters length.")
        if len(set(names)) != len(names):
            raise ValueError("FilterSet names must be unique.")
        object.__setattr__(self, "filters", filters)
        object.__setattr__(self, "names", names)

    def __len__(self) -> int:
        return len(self.filters)


def load_filter_set(names: Sequence[str]) -> FilterSet:
    """Load sedpy filters and keep the requested names as the band order.

    This is a small convenience for notebook-level experiments:

    ``filters = load_filter_set(["sdss_u0", "sdss_g0", "sdss_r0"])``

    The import is lazy so CompoSED itself remains importable without sedpy.
    """

    try:
        from sedpy.observate import load_filters
    except ImportError as exc:  # pragma: no cover - depends on optional sedpy.
        raise ImportError("load_filter_set requires sedpy. Install sedpy or pass a FilterSet directly.") from exc

    names = tuple(str(name) for name in names)
    return FilterSet(load_filters(list(names)), names=names)
