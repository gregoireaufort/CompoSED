# Version 0.1.1 Release Checklist

The release gate is intentionally split into software checks and scientific
checks. Passing unit tests does not establish posterior calibration.

## Software

- [x] `python -m pytest -q` passes in the full development environment.
- [ ] Core tests pass on Python 3.10, 3.11, and 3.12 without optional engines.
- [ ] Stable MAF and sampler adapter jobs pass in CI.
- [x] GitHub Actions defines core, sampler, SBI, notebook-hygiene, and isolated
  wheel-install release gates.
- [x] Wheel and source distributions pass `twine check`.
- [x] Wheel and source distributions installed in clean environments pass
  `pip check` and import `composed` and `inftools` outside the source tree.
- [x] The FSPS and CIGALE environment checks pass where those engines are
  installed.
- [x] Tutorial notebooks contain no execution errors or duplicate cell IDs.
- [x] Executed stable notebooks have been passed through
  `python scripts/sanitize_notebook_outputs.py --check`.
- [x] Generated catalogs, checkpoints, figures, external grids, and manuscript
  scratch files are not committed.
- [x] Saved inference products and model grids verify content hashes and
  provenance by default.

## Scientific

- [x] Mass scaling is applied exactly once and `log10_mass` means present-day
  surviving stellar mass for every stable backend.
- [x] FSPS photometry agrees with the independent direct FSPS/sedpy reference.
- [x] Named SFHs produce finite, non-negative histories and respect the
  age-of-the-Universe constraint.
- [x] Invalid physical support maps to `-inf` for deterministic likelihoods and
  cannot silently condition an SBI training prior.
- [x] The analytic bounded-Gaussian MAF validation passes.
- [ ] Each released real-backend MAF example reports held-out rank and coverage
  diagnostics generated from its own simulator and noise distribution.
- [x] Reference Monte Carlo overlays are labelled as finite reference
  calculations, not ground truth.

## Release

- [x] `CHANGELOG.md` describes the stable and experimental scope.
- [ ] The working tree is clean.
- [ ] The release commit is pushed and CI is green.
- [ ] Tag `v0.1.1` points to that release commit.
