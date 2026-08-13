#!/usr/bin/env python
"""Replace machine-local paths in saved notebook outputs.

The scientific source cells are left untouched. This script only sanitizes
textual output produced by an executed notebook, while preserving figures and
numerical results.

Examples
--------
Sanitize all stable tutorial and validation notebooks:

    python scripts/sanitize_notebook_outputs.py

Check whether sanitization would change a notebook:

    python scripts/sanitize_notebook_outputs.py --check notebook.ipynb
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def output_path_replacements() -> dict[str, str]:
    """Map paths visible to this interpreter onto auditable placeholders."""

    replacements = {
        str(REPO_ROOT): "${REPO_ROOT}",
        str(Path(sys.prefix).resolve()): "${PYTHON_ENV}",
    }
    for variable in ("SPS_HOME", "TMPDIR"):
        value = os.environ.get(variable)
        if value:
            path = Path(value).expanduser()
            placeholder = f"${{{variable}}}"
            replacements[str(path)] = placeholder
            replacements[str(path.resolve())] = placeholder

    spec = importlib.util.find_spec("pcigale")
    if spec is not None and spec.origin:
        source_root = Path(spec.origin).resolve().parents[1]
        replacements[str(source_root)] = "${CIGALE_SOURCE}"
    return replacements


def replace_text(value, replacements: dict[str, str]):
    """Recursively replace paths in JSON-compatible notebook output values."""

    if isinstance(value, str):
        for original, placeholder in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
            value = value.replace(original, placeholder)
        return value
    if isinstance(value, list):
        return [replace_text(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: replace_text(item, replacements) for key, item in value.items()}
    return value


def sanitize_notebook(path: Path, replacements: dict[str, str], *, write: bool) -> bool:
    """Sanitize one notebook and return whether its saved outputs changed."""

    payload = json.loads(path.read_text())
    original = json.dumps(payload, sort_keys=True)
    for cell in payload.get("cells", []):
        if "outputs" in cell:
            cell["outputs"] = replace_text(cell["outputs"], replacements)
    changed = json.dumps(payload, sort_keys=True) != original
    if changed and write:
        path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n")
    return changed


def default_notebooks() -> list[Path]:
    """Return the stable notebooks included in source distributions."""

    return sorted((REPO_ROOT / "notebooks" / "tutorials").glob("*.ipynb")) + sorted(
        (REPO_ROOT / "notebooks" / "validation").glob("*.ipynb")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="*", type=Path, help="Notebook paths; defaults to stable notebooks.")
    parser.add_argument("--check", action="store_true", help="Report unsanitized outputs without writing files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = args.notebooks or default_notebooks()
    replacements = output_path_replacements()
    changed = []
    for path in paths:
        if sanitize_notebook(path, replacements, write=not args.check):
            changed.append(path)
            print(f"{'needs sanitizing' if args.check else 'sanitized'}: {path}")
    return 1 if args.check and changed else 0


if __name__ == "__main__":
    raise SystemExit(main())
