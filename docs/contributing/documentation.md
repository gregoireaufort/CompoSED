# Building The Documentation

Install the documentation dependencies from the repository root:

```bash
python -m pip install -e ".[docs]"
```

Build HTML with warnings treated as errors:

```bash
sphinx-build -W --keep-going -b html docs docs/_build/html
```

The default build is fully offline. Set `COMPOSED_DOCS_INTERSPHINX=1` to add
links into the Python, NumPy, and Astropy API inventories when network access is
available.

or:

```bash
make -C docs html
```

Open `docs/_build/html/index.html` locally after a successful build.

## Writing rules

- User-guide pages follow scientific workflow, not source-tree order.
- Public docstrings use NumPy style.
- Array shapes, units, mask meaning, normalization, and failure behavior are
  explicit.
- Optional dependencies are not imported at documentation-build time merely to
  render an API page.
- Long tutorials are linked, not executed by Sphinx.
- New stable public names must appear in the curated API reference.
- New experimental features must be labeled in both the guide and API page.

## Documentation checks

The documentation CI job:

1. installs only the core package plus the `docs` extra;
2. imports `composed` and `inftools`;
3. builds Sphinx with `-W --keep-going`;
4. fails on broken internal references or autodoc import errors.

Backend-specific installation and numerical validation remain separate because
FSPS grids and CIGALE databases are external scientific dependencies.
