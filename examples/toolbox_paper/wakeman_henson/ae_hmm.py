"""Wakeman-Henson: AE-HMM Pipeline.

This script contains the code used to create Figure 2.
"""

from sys import argv

if len(argv) != 2:
    print("Please pass the run id, e.g. python ae_hmm.py 1")
    exit()
id = argv[1]

import mne
import pickle
import numpy as np
from glob import glob

from osl_dynamics import run_pipeline
from osl_dynamics.analysis import statistics
from osl_dynamics.inference import modes
from osl_dynamics.utils import plotting


def epoch_state_time_course(stc, tmin=-0.1, tmax=1.5):
    """Get subject-specific evoked responses in the state occupancies."""

    # Parcellated data files
    parc_files = sorted(glob("src/*/sflip_parc-raw.fif"))

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
            n_window=5,  # this should be what was used to prepare the training data
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
    epochs = epochs.reshape(19, 6, 401, 8)
    epochs = np.mean(epochs, axis=1)  # average over runs

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
    alp = pickle.load(open(inf_params_dir + "/alp.pkl", "rb"))
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
        y_label="State Activation",
        filename=plots_dir + "/epoched_stc.png",
    )


# Full pipeline
config = """
    load_data:
        data_dir: training_data
        data_kwargs:
            sampling_frequency: 250
            mask_file: MNI152_T1_8mm_brain.nii.gz
            parcellation_file: fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz
            n_jobs: 16
        prepare_kwargs:
            amplitude_envelope: True
            n_window: 5
    train_hmm:
        config_kwargs:
            n_states: 8
            learn_means: True
            learn_covariances: True
    calc_subject_ae_hmm_networks: {}
    plot_group_ae_networks:
        power_save_kwargs:
            plot_kwargs: {views: [lateral]}
    plot_alpha:
        kwargs: {n_samples: 2000}
    plot_summary_stats: {}
    plot_evoked_response:
        n_perm: 1000
        metric: copes
        significance_level: 0.05
"""

# Run analysis
run_pipeline(config, output_dir=f"results/run{id}", extra_funcs=[plot_evoked_response])
