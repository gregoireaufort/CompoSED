# Experimental Conditional Diffusion SBI

`inftools.experimental.diffusion` implements a masked conditional diffusion
model for array-based simulation-based inference.  It is experimental: useful
for research and notebooks, but less stable than the top-level MAF interface.

Diffusion is excluded from the stable CompoSED `0.1` top-level API. Experimental
high-level use imports `Diffusion` from `composed.sbi`; the class documented
below is the lower-level score-model implementation.

## Data Entering

Training data is a two-dimensional feature table:

```python
x_train.shape == (n_objects, n_features)
```

For generic precomputed arrays the feature vector can contain any continuous
quantities, for example:

```text
[mags, mag_errors, physical_parameters]
```

For Problem-driven photometric SBI, CompoSED constructs a stricter observation
vector in band order:

```text
[photometric context,
 uncertainty context,
 availability,
 censoring flag,
 limit-known flag,
 upper-limit depth,
 prior-transformed parameters]
```

The default photometric context is measured signal-to-noise plus
`log10(sigma / reference_flux)`. Fluxes and uncertainties remain in the
`SEDDataset.flux_unit` declared by the user. A censored flux is never supplied
to the network: its photometric feature is neutral, while its uncertainty,
limit, and state flags remain visible.

`FeatureMetadata` records the column names and scientific groups:

```python
from inftools.experimental.diffusion import FeatureMetadata

meta = FeatureMetadata.from_groups({
    "mags": ["u", "g", "r", "i", "z", "Y", "J", "H"],
    "params": ["redshift", "log10_mass"],
})
```

## Transformations

The estimator fits an internal standardizer on `x_train`.  Training then uses
standardized features:

1. draw a boolean condition mask;
2. keep `known = x0 * mask`;
3. add Gaussian diffusion noise to `x0`;
4. train the score network to predict `-epsilon`;
5. optionally compute loss only on unknown coordinates.

The fitted sampler returns samples in the original physical units.

## Masks And Cuts

Two masks have different scientific meanings:

- the **availability/censoring channels** are values in the observed data;
- the **diffusion mask** is a temporary training or inference instruction.

The diffusion-mask convention is:

```python
mask == True   # known / conditioned / clamped
mask == False  # unknown / sampled
```

Known entries are reclamped during every reverse-diffusion step.  This is the
core inpainting rule: the model can sample any missing subset while respecting
whatever photometry or parameters are fixed.

If both `mags` and `magerrs` groups exist, magnitude-error masks can be tied to
magnitude masks.  That prevents a hidden flux from leaking through a visible
catalog uncertainty.

For CompoSED photometry, the photometry, uncertainty, availability, censoring,
and upper-limit channels are tied by band during random masking. Hiding a band
therefore hides its complete observed state, while the stored availability
flag itself remains distinct from this artificial training mask.

## Normalization

Backend mass normalization is applied upstream by `Problem.simulate`, exactly
as it is in the deterministic likelihood. Before diffusion training, bounded
continuous priors are mapped to unconstrained neural coordinates. Returned
samples are inverse-transformed to the physical parameter units declared by
the `ParameterSpace`, so a bounded mass or redshift cannot leak outside its
prior merely because of neural sampling noise.

## Important Functions To Audit

- `FeatureMetadata.from_groups`: feature order and group definitions.
- `SBITrainingSet.diffusion_observation_features`: measured values and
  availability/censoring channels.
- `make_training_mask`: which data are visible during training.
- `ConditionalDiffusionEstimator.fit`: the diffusion noise and score loss.
- `ConditionalDiffusionEstimator.sample`: standardization, mask handling, and
  known-feature reclamping.

## Minimal Example

For external paired arrays, the high-level route is:

```python
from composed import SBITrainingSet, train_sbi
from composed.sbi import Diffusion

training = SBITrainingSet.from_arrays(
    theta_train,
    x_train,
    theta_names=["z", "log10_mass"],
    x_names=["g", "r"],
    source="presampled_forward_model",
)
posterior = train_sbi(training, Diffusion(epochs=20, batch_size=128), seed=1)
samples = posterior.sample(x_obs, num_samples=256, steps=50)
```

The lower-level equivalent, useful when developing new masks or score models,
is:

```python
import numpy as np

from inftools.experimental.diffusion import ConditionalDiffusionEstimator, FeatureMetadata

rng = np.random.default_rng(1)
z = rng.uniform(0.0, 2.0, size=512)
mass = rng.normal(10.0, 0.3, size=512)
g = 22.0 + z - 0.15 * (mass - 10.0)
r = 21.7 + 0.6 * z - 0.12 * (mass - 10.0)
x_train = np.column_stack([g, r, z, mass])

meta = FeatureMetadata.from_groups({
    "mags": ["g", "r"],
    "params": ["z", "log10_mass"],
})

estimator = ConditionalDiffusionEstimator(meta, model="mlp", hidden_features=64)
estimator.fit(
    x_train,
    mask_config={"unknown_fraction": {"mags": 0.0, "params": 1.0}},
    epochs=20,
    batch_size=128,
)

known = np.array([[g[0], r[0], np.nan, np.nan]])
mask = np.array([[True, True, False, False]])
samples = estimator.sample(known, mask, num_samples=256, steps=50)

print(np.median(samples[0, :, 2:], axis=0))
```

## Device And Dtype Policy

Diffusion training is meant to run on accelerators when they are available.
Use the default device selection unless you have a reason not to:

```python
estimator = ConditionalDiffusionEstimator(meta, device="auto")
```

`device="auto"` tries CUDA, then Apple MPS, then CPU.  Before committing to a
device, CompoSED runs a tiny float32 forward/backward smoke test.  This catches
common cases where an accelerator is nominally available but cannot run the
operations needed by the score model.

All torch tensors created by the diffusion estimator are float32, even if a
notebook has changed PyTorch's global default dtype to float64.  This is
important on Apple MPS, which does not support float64 training tensors.

Useful explicit options:

```python
ConditionalDiffusionEstimator(meta, device="cuda")
ConditionalDiffusionEstimator(meta, device="mps")
ConditionalDiffusionEstimator(meta, device="cpu")
ConditionalDiffusionEstimator(meta, device="mps", allow_device_fallback=False)
```

With fallback enabled, a failing requested accelerator can fall back to CPU with
a warning.  With fallback disabled, failure raises a clear error before
training starts.

## Sanity Checks

Start with data simulated from the same distribution used for training.  Check
that known photometric entries are exactly clamped in returned physical units,
that censored values are represented only by their limits, that all returned
parameters remain inside prior support, that posterior summaries recover the
true parameters statistically, and that held-out feature inpainting has
sensible residuals before moving to real data.
