---
orphan: true
---

# osl-dynamics

> osl-dynamics is a Python toolbox for studying brain dynamics using neuroimaging data (MEG, EEG and fMRI). It provides generative models that decompose data into dynamic functional brain networks, often called brain states or modes, including the Hidden Markov Model (HMM) and Dynamic Network Modes (DyNeMo). It also provides everything needed for a complete analysis: M/EEG preprocessing, source reconstruction and parcellation, data preparation, spectral estimation, power maps, functional connectivity, burst detection, simulation and statistical significance testing.

osl-dynamics was developed by the OHBA Methods Group at the University of Oxford. It works with MNE-Python: a typical M/EEG workflow preprocesses, source reconstructs and parcellates the data (all available in `osl_dynamics.meeg`), then models the dynamics of the parcel time courses. Data can be loaded from NumPy (.npy), MATLAB (.mat), text (.txt) or MNE (.fif) files.

If you use osl-dynamics, please cite: Gohil, C., Huang, R., Roberts, E., van Es, M. W., Quinn, A. J., Vidaurre, D., and Woolrich, M. W. (2024). osl-dynamics, a toolbox for modeling fast dynamic brain activity. eLife, 12, RP91949.

## Installation

- conda (recommended, includes TensorFlow): `mamba create -n osld -c conda-forge osl-dynamics`
- pip without TensorFlow: `pip install osl-dynamics`
- pip with TensorFlow: `pip install "osl-dynamics[tf]"`, or `pip install "osl-dynamics[tf-cuda]"` for an NVIDIA GPU on Linux

TensorFlow is only needed to train models (`osl_dynamics.models`). Data preparation, M/EEG processing, spectral analysis, simulation and statistics work without it.

## Core workflow

A typical M/EEG analysis has five steps. For fMRI or sensor-level data, start at step 3 and prepare the data with `{"standardize": {}}` only (fMRI HMMs use `learn_means=True`).

### 1. M/EEG processing (each session)

```python
import mne
from osl_dynamics.meeg import preproc, rhino, source_recon, parcellation, Session

# Sensor-level preprocessing with MNE and osl-dynamics
raw = mne.io.read_raw_fif("sub-01_task-rest.fif", preload=True)
raw = raw.filter(l_freq=1, h_freq=45).resample(sfreq=250)
raw = preproc.detect_bad_segments(raw, picks="mag")
raw = preproc.detect_bad_channels(raw, picks="mag")
raw.save("derivatives/preprocessed/sub-01_preproc-raw.fif")

# Surface extraction (once per subject), coregistration and forward model
rhino.extract_surfaces(mri_file="sub-01_T1w.nii.gz", outdir="derivatives/anat_surfaces/sub-01")
session = Session(
    outdir="derivatives/osl",
    id="sub-01_task-rest",
    preproc_file="derivatives/preprocessed/sub-01_preproc-raw.fif",
    surfaces_dir="derivatives/anat_surfaces/sub-01",
)
rhino.extract_fiducials_and_headshape_from_fif(session)
rhino.coregister_head_and_mri(session)
rhino.forward_model(session, model="Single Layer", gridstep=8)

# LCMV beamformer, then parcellation with symmetric orthogonalization
source_recon.lcmv_beamformer(session, chantypes=["mag", "grad"], rank={"meg": 60})
parcel_data = parcellation.parcellate_lcmv(
    session,
    parcellation_file="atlas-Glasser_nparc-52_space-MNI_res-8x8x8.nii.gz",
    orthogonalization="symmetric",
)
parcellation.save_as_fif(parcel_data, session.preproc_file, filename="derivatives/osl/sub-01_task-rest/lcmv-parc-raw.fif")
```

`osl_dynamics.meeg.parallel.run` runs a function like this on many sessions in parallel. Without a subject MRI, skip `rhino.extract_surfaces`, use `osl_dynamics.files.mni152_surfaces.directory` as `surfaces_dir` and pass `allow_mri_scaling=True` to `rhino.coregister_head_and_mri`.

### 2. Sign flipping (source-space M/EEG)

```python
from osl_dynamics.data import sign_flipping

# Align the sign of the parcel time courses across sessions
# (parc_fifs are the parcellated fif files from step 1)
template_cov, _ = sign_flipping.build_template_mne_raw(parc_fifs)
for parc_fif, sflip_fif in zip(parc_fifs, sflip_fifs):
    sign_flipping.sign_flip_mne_raw(parc_fif, template_cov=template_cov, output_file=sflip_fif)
```

### 3. Model training

```python
from osl_dynamics.data import Data
from osl_dynamics.models.hmm import Config, Model

# Load parcel time courses: fif files, a directory, a list of files or a list of arrays
data = Data(sflip_fifs, picks="misc", reject_by_annotation="omit", sampling_frequency=250)

# Time-delay embedding + PCA, then standardize
# (n_pca_components is a round number roughly twice the number of channels, e.g. 120 for 52 parcels)
data.prepare({
    "tde_pca": {"n_embeddings": 15, "n_pca_components": 120},
    "standardize": {},
})

# Train a TDE-HMM
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
model.save("results/model")  # reload with osl_dynamics.models.load("results/model")

# State probabilities: a list with one (n_samples, n_states) array per session
alpha = model.get_alpha(data)
```

### 4. Post-hoc analysis

```python
from osl_dynamics.analysis import spectral, power, connectivity, post_hoc
from osl_dynamics.inference import modes

parcellation_file = "atlas-Glasser_nparc-52_space-MNI_res-8x8x8.nii.gz"

# State spectra from the time series before preparation; a Data object is
# aligned to alpha automatically (or use data.trim_time_series with arrays)
spectra = spectral.multitaper_spectra(data, alpha=alpha, frequency_range=[1, 45])
group = spectra.group_average()
p = group.power_maps()                         # (n_states, n_parcels), change from static power
c = group.coherence_networks(percentile=97)    # (n_states, n_parcels, n_parcels)
power.save(p, parcellation_file=parcellation_file, filename="power_.png")
connectivity.save(c, parcellation_file=parcellation_file, filename="coh_.png")

# Summary statistics
stc = modes.argmax_time_courses(alpha)
fo = post_hoc.fractional_occupancies(stc)  # (n_sessions, n_states)
lt = post_hoc.mean_lifetimes(stc, sampling_frequency=250)
intv = post_hoc.mean_intervals(stc, sampling_frequency=250)
sr = post_hoc.switching_rates(stc, sampling_frequency=250)
```

### 5. Statistical significance testing

```python
from osl_dynamics import stats

# Group difference in each summary statistic, corrected across states
# (assignments is a group label for each session)
for name, x in {"fo": fo, "lt": lt, "intv": intv, "sr": sr}.items():
    result = stats.group_diff(x, assignments, n_perm=1000)
    print(name, result.pvalues, result.effect_size())

# Group difference in the state power maps, corrected across states and parcels
session_p = spectra.power_maps()  # (n_sessions, n_states, n_parcels)
result = stats.group_diff(session_p, assignments, n_perm=1000)
```

The Config API runs the same pipeline from a YAML file: `osl-dynamics config.yml results`, or `osl_dynamics.run_pipeline(config, output_dir="results")`. Each top-level key is a function in `osl_dynamics.config_api.wrappers`.

## API notes

- `Data` keeps data in memory by default and writes nothing to disk. With `load_memmaps=True` it writes memory maps to a subdirectory of `store_dir` (default `./tmp`), which is deleted automatically when the script ends. Don't call `data.delete_dir()` at the end of a script; it is only needed to free disk space early when using `load_memmaps=True` in a long-running session such as a notebook.
- Every model module (`osl_dynamics.models.hmm`, `.dynemo`, `.dyneste`, `.mdynemo`, `.hive`, `.dive`, `.hmm_poi`, `.sc_dynemo`) defines a `Config` dataclass and a `Model` class with `fit`, `get_alpha`, `save` and initialization methods. Models are loaded with `osl_dynamics.models.load`.
- HMM-family models infer mutually exclusive states (use `spectral.multitaper_spectra`); DyNeMo-family models infer mixtures of modes (use `spectral.regression_spectra`).
- `spectral.welch_spectra`, `spectral.multitaper_spectra` and `spectral.regression_spectra` return `StateSpectra` or `RegressionSpectra` objects (`StaticSpectra` without `alpha`, as do `analysis.static.welch_spectra` and `analysis.static.multitaper_spectra`). They have the attributes `frequencies`, `psd` and `coherence` (and `psd_change` and `coherence_change` for state and regression spectra), and the methods `group_average`, `power_maps`, `coherence_networks`, `decompose`, `plot_psd` and `save`; load with `spectral.Spectra.load`. Arguments after `alpha` are keyword-only.
- `osl_dynamics.stats` has the ready-made tests `evoked_response`, `paired_diff`, `group_diff`, `condition_diff` and `regression`, each returning a `TestResult` with `pvalues` and `effect_size()`. Custom designs use `DesignConfig`, `Design` and `GLM` with `MaxStatPermutation` or `ClusterPermutation`. Corrections: `correction="max_stat"` (default), cluster-based, TFCE, or `stats.fdr` / `stats.holm` / `stats.bonferroni` on p-values.
- Simulations compose a state model with an observation model: `simulation.Simulation(state_model=simulation.HMM(...), obs_model=simulation.MVN(...), n_samples=...)`. State models: `HMM`, `HSMM`, `HierarchicalHMM`, `MDynHMM`, `MixedSine`, `MixedHSMM`. Observation models: `MVN`, `OscillatoryMVN`, `MDyn_MVN`, `MSess_MVN`, `MAR`, `TDECovs`, `OscillatoryBursts`, `Poisson`, `Dirichlet`. Classes such as `HMM_MVN` and `HSMM_MVN` combine common pairs.
- M/EEG processing (`osl_dynamics.meeg`): `preproc`, `rhino` (surface extraction, coregistration, forward model), `source_recon` (LCMV beamformer), `parcellation` (with `orthogonalization="symmetric"`), and `Session` for the files of a session. `preproc.ica_label` only removes artefacts from the channels it is fitted on: use `picks="meg"` for Elekta data so the magnetometers and gradiometers are both cleaned.
- Parcellations, masks and surfaces are downloaded from the osl-files repository on first use and cached; refer to them by file name. `osl_dynamics.files.fetch_file`, `fetch_directory` and `fetch_all` download files explicitly (e.g. before working offline).
- Spelling is American throughout (e.g. `standardize`, `orthogonalization`, `regularization`).

## Removed APIs

Code written for earlier versions may use names that no longer exist:

- `osl_dynamics.glm` and `osl_dynamics.analysis.statistics` are replaced by `osl_dynamics.stats`.
- Spectra functions no longer return tuples such as `f, psd, coh = spectral.multitaper_spectra(...)`; they return a spectra object with `.frequencies`, `.psd` and `.coherence`.
- The `use_tfrecord` argument and the `mask_file` argument are removed.
- osl-ephys is no longer required for M/EEG preprocessing and source reconstruction; use `osl_dynamics.meeg`.

## Documentation

Every page is available as Markdown by appending `.md` to its URL.

- [Full documentation](https://osl-dynamics.readthedocs.io/en/latest/llms-full.txt): the user guide, FAQ and tutorials as one Markdown file (the API reference is not included).
- [Installation](https://osl-dynamics.readthedocs.io/en/latest/install/index.html): conda and pip installation, with separate pages for Windows and the Oxford BMRC cluster.
- [Getting started](https://osl-dynamics.readthedocs.io/en/latest/getting_started.html): a typical workflow, a minimal HMM example and the Config API.
- [M/EEG processing](https://osl-dynamics.readthedocs.io/en/latest/user_guide/meeg.html): preprocessing, surface extraction, coregistration, source reconstruction, parcellation, orthogonalization and sign flipping.
- [Parcellations](https://osl-dynamics.readthedocs.io/en/latest/user_guide/parcellations/index.html): the available parcellation files.
- [Models](https://osl-dynamics.readthedocs.io/en/latest/user_guide/models/index.html): HMM, DyNeMo, DyNeStE, M-DyNeMo and HIVE, with figures.
- [Spectral analysis](https://osl-dynamics.readthedocs.io/en/latest/user_guide/spectra.html): which spectra function to use for each model, the spectra objects, power maps and coherence networks.
- [Statistical significance testing](https://osl-dynamics.readthedocs.io/en/latest/user_guide/stats.html): which test and which correction for multiple comparisons (maximum statistic, cluster-based, TFCE, FDR) to use for each kind of output, covariates and effect sizes.
- [Tutorials](https://osl-dynamics.readthedocs.io/en/latest/tutorials.html): step by step walkthroughs covering M/EEG processing, data loading and preparation, time-delay embedding, static analysis, HMM and DyNeMo training, spectra, power maps, connectivity, summary statistics, task analysis and statistical testing.
- [FAQ](https://osl-dynamics.readthedocs.io/en/latest/faq.html): guidance on preprocessing, time-delay embedding, choosing parameters, choosing between runs and interpreting results.
- [API reference](https://osl-dynamics.readthedocs.io/en/latest/autoapi/index.html): documentation for every module, class and function.

## Examples

- [Example scripts](https://github.com/OHBA-analysis/osl-dynamics/tree/main/examples): full pipelines for MEG and EEG preprocessing, MEG and fMRI analysis, spectra, static analysis, statistics, simulation, decoding, TINDA and plotting.

## Source and community

- [GitHub repository](https://github.com/OHBA-analysis/osl-dynamics)
- [PyPI package](https://pypi.org/project/osl-dynamics/)
- [conda-forge package](https://anaconda.org/conda-forge/osl-dynamics)
- [Paper in eLife](https://elifesciences.org/articles/91949)
- [Issue tracker](https://github.com/OHBA-analysis/osl-dynamics/issues)
