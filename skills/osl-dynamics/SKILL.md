---
name: osl-dynamics
description: Analyze brain dynamics in MEG, EEG and fMRI data with the osl-dynamics Python toolbox (import osl_dynamics). Use when writing, running or debugging code that uses osl-dynamics, including M/EEG preprocessing, source reconstruction, parcellation and sign flipping, time-delay embedding, training a Hidden Markov Model (HMM), DyNeMo, DyNeStE, M-DyNeMo or HIVE, state/mode spectra, power maps, coherence networks, summary statistics such as fractional occupancy and lifetimes, and permutation tests.
license: Apache-2.0
---

# osl-dynamics

osl-dynamics infers dynamic functional brain networks (states or modes) from MEG, EEG and fMRI data and provides the full analysis around the models: M/EEG processing, data preparation, spectral analysis, summary statistics, plotting and statistical significance testing.

## Documentation

- [llms.txt](https://osl-dynamics.readthedocs.io/en/latest/llms.txt): overview, the full workflow as code and the API changes.
- [llms-full.txt](https://osl-dynamics.readthedocs.io/en/latest/llms-full.txt): the user guide, FAQ and tutorials in one Markdown file.
- Every docs page is available as Markdown by appending `.md` to its URL, including the API reference (`.../autoapi/osl_dynamics/<module>/index.html.md`).
- [Example scripts](https://github.com/OHBA-analysis/osl-dynamics/tree/main/examples)

When unsure about an argument, inspect the installed version (`help(func)` or `inspect.signature(func)`) instead of guessing.

## Workflow

1. **M/EEG processing** (`osl_dynamics.meeg`, per session): MNE filtering (1-45 Hz) and resampling (250 Hz), `preproc.detect_bad_segments` / `preproc.detect_bad_channels`, `rhino.extract_surfaces`, `rhino.extract_fiducials_and_headshape_from_fif`, `rhino.coregister_head_and_mri`, `rhino.forward_model`, `source_recon.lcmv_beamformer`, `parcellation.parcellate_lcmv(..., orthogonalization="symmetric")`, `parcellation.save_as_fif`. A `meeg.Session` holds the files of a session; `meeg.parallel.run` processes many sessions. See `examples/meg_preproc`.
2. **Sign flipping** (source-space M/EEG only): `osl_dynamics.data.sign_flipping.build_template_mne_raw(fifs)` then `sign_flip_mne_raw(fif, template_cov=..., output_file=...)` for each session.
3. **Training**: load, prepare, train, get `alpha`.
4. **Post-hoc analysis**: spectra, power maps, coherence networks, summary statistics.
5. **Statistics**: `osl_dynamics.stats`.

```python
from osl_dynamics.data import Data
from osl_dynamics.models.hmm import Config, Model
from osl_dynamics.analysis import spectral, power, connectivity, post_hoc
from osl_dynamics.inference import modes
from osl_dynamics import stats

# sflip_fifs are the sign-flipped parcellated fif files
data = Data(sflip_fifs, picks="misc", reject_by_annotation="omit", sampling_frequency=250)
data.prepare({"tde_pca": {"n_embeddings": 15, "n_pca_components": 120}, "standardize": {}})

config = Config(
    n_states=8,
    n_channels=data.n_channels,
    sequence_length=200,
    learn_means=False,
    learn_covariances=True,
    batch_size=256,
    learning_rate=0.01,
    n_epochs=20,
)
model = Model(config)
model.random_state_time_course_initialization(data, n_init=3, n_epochs=1)
model.fit(data)
alpha = model.get_alpha(data)  # list of (n_samples, n_states) arrays

spectra = spectral.multitaper_spectra(data, alpha=alpha, frequency_range=[1, 45])
group = spectra.group_average()
parcellation_file = "atlas-Glasser_nparc-52_space-MNI_res-8x8x8.nii.gz"
power.save(group.power_maps(), parcellation_file=parcellation_file, filename="pow_.png")
connectivity.save(group.coherence_networks(percentile=97), parcellation_file=parcellation_file, filename="coh_.png")

stc = modes.argmax_time_courses(alpha)
fo = post_hoc.fractional_occupancies(stc)
result = stats.group_diff(fo, assignments, n_perm=1000)
print(result.pvalues, result.effect_size())
```

## Choosing a model

| Model | Module | Use when |
|---|---|---|
| HMM | `models.hmm` | Default. Mutually exclusive states, interpretable summary statistics, fast, no GPU needed. |
| DyNeMo | `models.dynemo` | Networks overlap in time (a linear mixture of modes), e.g. task responses. A GPU helps. |
| DyNeStE | `models.dyneste` | States with long-range temporal dependencies. |
| M-DyNeMo | `models.mdynemo` | Power and functional connectivity change independently. |
| HIVE | `models.hive` | Sessions differ systematically (subjects, scanners, sites). |

Every model module defines `Config` and `Model`. Training needs TensorFlow (`pip install "osl-dynamics[tf]"` or the conda-forge package); everything else works without it.

## Recommended settings

- **Source-space M/EEG**: sign flip, then `tde_pca` with `n_embeddings` giving a ~60 ms window (15 at 250 Hz, 7 at 100 Hz) and `n_pca_components` at least twice the number of parcels, then `standardize`. Use `learn_means=False`.
- **Sensor-level M/EEG or fMRI**: `{"standardize": {}}` only. fMRI HMMs use `learn_means=True`.
- **Amplitude envelope**: `{"amplitude_envelope": {}, "moving_average": {"n_window": 5}, "standardize": {}}` with `learn_means=True`.
- **Number of states/modes**: 6-14, start with 8. Check that conclusions hold for nearby values.
- **Initialization**: HMM `random_state_time_course_initialization(data, n_init=3, n_epochs=1)`; DyNeMo `random_subset_initialization(data, n_init=5, n_epochs=2)`.
- **DyNeMo config**: `n_modes`, `sequence_length=100`, `inference_n_units=64`, `inference_normalization="layer"`, `model_n_units=64`, `model_normalization="layer"`, `learn_alpha_temperature=True`, `initial_alpha_temperature=1.0`, `do_kl_annealing=True`, `kl_annealing_curve="tanh"`, `kl_annealing_sharpness=5`, `n_kl_annealing_epochs=10`, `batch_size=256`, `learning_rate=0.01`, `n_epochs=20`.
- **Run-to-run variability**: train several runs and keep the lowest `model.free_energy(data)`, or choose a representative run with `modes.select_run`.
- **Reproducibility**: `osl_dynamics.utils.set_random_seed(seed)` before building the model.
- **Large datasets**: `Data.save_tfrecord_dataset` with `osl_dynamics.data.load_tfrecord_dataset`.

## Post-hoc analysis

- HMM-family (states): `spectral.multitaper_spectra`. DyNeMo-family (modes): `spectral.regression_spectra` with the raw `alpha`.
- Spectra are calculated from the data before preparation. Passing the prepared `Data` object aligns it to `alpha` automatically; with arrays, use `data.trim_time_series(n_embeddings=..., sequence_length=..., prepared=False)`.
- Spectra objects have `frequencies`, `psd` and `coherence` (and `psd_change` and `coherence_change` for state and regression spectra) and the methods `group_average`, `power_maps`, `coherence_networks`, `decompose`, `plot_psd`, `save` (`spectral.Spectra.load` to reload). Power maps and coherence networks default to the change from the static (time-averaged) value.
- HMM summary statistics from `stc = modes.argmax_time_courses(alpha)`: `post_hoc.fractional_occupancies`, `mean_lifetimes`, `mean_intervals`, `switching_rates` (the last three need `sampling_frequency`).
- DyNeMo summary statistics use `modes.reweight_alphas(alpha, covs)` with `covs = model.get_covariances()`, e.g. the time-averaged reweighted mixing coefficient of each mode.
- Parcellations, masks and surfaces are referred to by file name and downloaded on first use.

## Statistics

`osl_dynamics.stats` tests take an array with subjects first and any number of target dimensions, and return a `TestResult` with `pvalues` and `effect_size()`.

| Question | Function |
|---|---|
| Is a response different from zero? | `stats.evoked_response` |
| Is a paired difference different from zero? | `stats.paired_diff` |
| Do groups differ? | `stats.group_diff` |
| Do conditions within subjects differ? | `stats.condition_diff` |
| Does a quantity change with a covariate? | `stats.regression` |
| Custom contrasts | `stats.DesignConfig` with `stats.MaxStatPermutation` or `stats.ClusterPermutation` |

Corrections (`correction=`): `"max_stat"` (default) for states, parcels and summary statistics; `"cluster"` or `"tfce"` with `cluster_dims` for effects over time or frequency; `"fdr"` for connectivity. All tests except `condition_diff` take covariates, e.g. `covariates={"age": age}`.

## Common mistakes

- `osl_dynamics.glm` and `osl_dynamics.analysis.statistics` no longer exist: use `osl_dynamics.stats`.
- Spectra functions return an object, not a tuple: `f, psd, coh = spectral.multitaper_spectra(...)` fails. Arguments after `alpha` are keyword-only.
- The `use_tfrecord` and `mask_file` arguments no longer exist.
- osl-ephys is not needed for M/EEG processing: use `osl_dynamics.meeg`.
- Calculating spectra from the TDE/PCA data instead of the original parcel time courses.
- Assuming `alpha` has the same length as the data: TDE drops `n_embeddings // 2` samples from each end and sequencing drops the incomplete sequence at the end.
- Learning means on TDE data (`learn_means=True`): the networks are in the covariances.
- Names use American spelling (`standardize`, `orthogonalization`, `regularization`).
