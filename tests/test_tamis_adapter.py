import pytest

import inftools.tamis_adapter as adapter


def test_external_tamis_dependency_error_is_helpful(monkeypatch):
    monkeypatch.setattr(adapter, "_HAS_TAMIS", False)
    monkeypatch.setattr(adapter, "_TAMIS_IMPORT_ERROR", ImportError("missing test dependency"))

    with pytest.raises(ImportError, match="composed.MixedTAMIS"):
        adapter.run_tamis(None, None)
