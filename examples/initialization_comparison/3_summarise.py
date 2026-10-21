"""Compare the initialization methods.

For each method and selection criterion: the final free energy over runs
(lower is better, and a smaller spread means the result depends less on
the run), the time spent initializing, and how reproducible the states
are across runs (networks found by match_runs in most runs).
"""

import glob
import pickle

import numpy as np
from osl_dynamics.inference import modes

results = {}
for path in sorted(glob.glob("results/*/run*/summary.pkl")):
    summary = pickle.load(open(path, "rb"))
    key = f"{summary['method']}_{summary['select']}"
    results.setdefault(key, []).append(summary)

print(
    f"{'method':36s} {'runs':>4s} {'FE median':>10s} {'FE best':>10s} "
    f"{'FE spread':>10s} {'init (s)':>9s} {'train (s)':>9s} {'networks in >=80% runs':>24s}"
)
for key, summaries in results.items():
    free_energy = np.array([s["free_energy"] for s in summaries])
    init_time = np.mean([s["init_time"] for s in summaries])
    train_time = np.mean([s["train_time"] for s in summaries])

    # Reproducibility: group the states of all runs into networks, using
    # the vectorised covariances as features
    covs = [np.load(f"results/{key}/run{s['run']:02d}/covs.npy") for s in summaries]
    features = [c.reshape(c.shape[0], -1) for c in covs]
    networks = modes.match_runs(features)
    n_runs = len(summaries)
    found_in = np.bincount(np.concatenate(networks))
    n_reliable = int(np.sum(found_in >= 0.8 * n_runs))

    print(
        f"{key:36s} {n_runs:4d} {np.median(free_energy):10.3f} "
        f"{free_energy.min():10.3f} {np.ptp(free_energy):10.3f} "
        f"{init_time:9.0f} {train_time:9.0f} {n_reliable:24d}"
    )
