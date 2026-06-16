from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class FilterSet:
    """Thin wrapper for backend filter objects, usually sedpy filters."""

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
        object.__setattr__(self, "filters", filters)
        object.__setattr__(self, "names", names)

    def __len__(self) -> int:
        return len(self.filters)
