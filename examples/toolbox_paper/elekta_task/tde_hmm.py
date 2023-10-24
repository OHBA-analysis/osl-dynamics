"""Elekta Task Dataset: TDE-HMM Pipeline.

In this script we train an Time-Delay Embedded Hidden Markov Model (TDE-HMM)
on source reconstructed task MEG data.

The examples/toolbox_paper/elekta_task/get_data.py script can be used
to download the training data.

Functions listed in the config are defined in osl_dynamics.config_api.wrappers.
"""

from sys import argv

if len(argv) != 2:
    print("Please pass the run id, e.g. python tde_hmm.py 1")
    exit()
id = int(argv[1])

import mne
import pickle
import numpy as np
from glob import glob

from osl_dynamics import run_pipeline
from osl_dynamics.analysis import statistics
from osl_dynamics.data import Data
from osl_dynamics.inference import modes
from osl_dynamics.utils import plotting


def epoch_state_time_course(stc, tmin=-0.1, tmax=1.5):
    """Get subject-specific evoked responses in the state occupancies."""

    # Parcellated data files
    parc_files = sorted(glob("data/src/*/sflip_parc-raw.fif"))

    # Epoch the state time courses
    event_id = {
        "famous/first": 5,
        "famous/immediate": 6,
        "famous/last": 7,
        "unfamiliar/first": 13,
        "unfamiliar/immediate": 14,
        "unfamiliar/last": 15,
        "scrambled/first": 17,
        "scrambled/immediate": 18,
        "scrambled/last": 19,
    }
    epochs_ = []
    for s, p in zip(stc, parc_files):
        raw = modes.convert_to_mne_raw(
            s,
            p,
            n_embeddings=15,  # this should be what was used to prepare the training data
        )
        events = mne.find_events(raw, min_duration=0.005, verbose=False)
        e = mne.Epochs(
            raw,
            events,
            event_id,
            tmin=tmin,
            tmax=tmax,
            verbose=False,
        )
        epochs_.append(e.get_data(picks="misc"))

    # Time axis (we need to correct for the 34 ms delay in the trigger)
    t = e.times - 34e-3

    # Calculate subject-specific averaged evoked responses
    epochs = []
    for e_ in epochs_:
        epochs.append(np.mean(e_, axis=0).T)
    epochs = np.array(epochs)

    # Baseline correct
    epochs -= np.mean(
        epochs[:, : int(abs(tmin) * raw.info["sfreq"])],
        axis=1,
        keepdims=True,
    )

    return t, epochs


def plot_evoked_response(data, output_dir, n_perm, metric, significance_level):
    """Perform evoked response analysis with state time courses."""

    # Directories
    inf_params_dir = f"{output_dir}/inf_params"
    plots_dir = f"{output_dir}/alphas"

    # Get inferred state time course
    alp = pickle.load(open(f"{inf_params_dir}/alp.pkl", "rb"))
    stc = modes.argmax_time_courses(alp)

    # Epoch and do stats
    t, epochs = epoch_state_time_course(stc)
    pvalues = statistics.evoked_response_max_stat_perm(
        epochs, n_perm=n_perm, metric=metric
    )

    # Plot epoched state time courses with significant time points highlighed
    plotting.plot_evoked_response(
        t,
        np.mean(epochs, axis=0),
        pvalues,
        significance_level=significance_level,
        labels=[f"State {i + 1}" for i in range(epochs.shape[-1])],
        x_label="Time (s)",
        y_label="State Probability",
        filename=f"{plots_dir}/epoched_stc.png",
    )

# Load data
data = Data(
    inputs=sorted(glob("data/src/*/sflip_parc-raw.fif")),
    picks="misc",
    reject_by_annotation="omit",
    sampling_frequency=250,
    mask_file="MNI152_T1_8mm_brain.nii.gz",
    parcellation_file="fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz",
    store_dir=f"tmp_{id:02d}",
    n_jobs=8,
)
data.prepare({
    "tde_pca": {"n_embeddings": 15, "n_pca_components": 80},
    "standardize": {},
})

# Full pipeline
config = """
    train_hmm:
        config_kwargs:
            n_states: 8
            learn_means: False
            learn_covariances: True
    multitaper_spectra:
        kwargs:
            frequency_range: [1, 45]
            n_jobs: 8
        nnmf_components: 2
    plot_group_nnmf_tde_hmm_networks:
        nnmf_file: spectra/nnmf_2.npy
        power_save_kwargs:
            plot_kwargs: {views: [lateral]}
    plot_alpha:
        kwargs: {n_samples: 2000}
    plot_hmm_network_summary_stats: {}
    plot_evoked_response:
        n_perm: 1000
        metric: copes
        significance_level: 0.05
"""

# Run analysis
run_pipeline(
    config,
    data=data,
    output_dir=f"results/run{id:02d}",
    extra_funcs=[plot_evoked_response],
)
