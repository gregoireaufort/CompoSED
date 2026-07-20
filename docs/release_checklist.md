# Version 0.1 Release Checklist

The release gate is intentionally split into software checks and scientific
checks. Passing unit tests does not establish posterior calibration.

## Software

- [ ] `python -m pytest -q` passes in the full development environment.
- [ ] Core tests pass on Python 3.10, 3.11, and 3.12 without optional engines.
- [ ] Stable MAF and sampler adapter jobs pass in CI.
- [ ] Wheel and source distributions pass `twine check`.
- [ ] A wheel installed outside the source tree imports `composed` and
  `inftools`.
- [ ] The FSPS and CIGALE environment checks pass where those engines are
  installed.
- [ ] Tutorial notebooks contain no execution errors or duplicate cell IDs.
- [ ] Generated catalogs, checkpoints, figures, external grids, and manuscript
  scratch files are not committed.

## Scientific

- [ ] Mass scaling is applied exactly once and `log10_mass` means present-day
  surviving stellar mass for every stable backend.
- [ ] FSPS photometry agrees with the independent direct FSPS/sedpy reference.
- [ ] Named SFHs produce finite, non-negative histories and respect the
  age-of-the-Universe constraint.
- [ ] The analytic bounded-Gaussian MAF validation passes.
- [ ] Each released real-backend MAF example reports held-out rank and coverage
  diagnostics generated from its own simulator and noise distribution.
- [ ] Reference Monte Carlo overlays are labelled as finite reference
  calculations, not ground truth.

## Release

- [ ] `CHANGELOG.md` describes the stable and experimental scope.
- [ ] The working tree is clean.
- [ ] The release commit is pushed and CI is green.
- [ ] Tag `v0.1.0` points to that release commit.
