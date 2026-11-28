"""osl-dynamics tasks for the MATLAB interface.

A job is a folder holding job.json, {"task": <name>, "args": {...}}, and
the arrays the arguments refer to. An array is stored as a raw binary file
in column-major (MATLAB) order and referred to as
{"ndarray": <file>, "shape": [...], "dtype": <dtype>}. The result is
written to result.json in the same folder, in the same format, with each
list written as {"list": [...]}. If the task fails, the traceback is
written to error.txt.

Usage: python osld_bridge.py <job folder>
"""

import gc
import importlib.metadata
import json
import math
import os
import platform
import re
import sys
import time
import traceback

import numpy as np


def run(job_dir):
    """Run the job in job_dir and write result.json."""
    job_dir = str(job_dir)
    with open(os.path.join(job_dir, "job.json"), encoding="utf-8") as f:
        job = json.load(f)
    args = _decode(job.get("args") or {}, job_dir)
    args = {k: v for k, v in args.items() if v is not None}
    out = TASKS[job["task"]](store_dir=os.path.join(job_dir, "store"), **args)
    result = _Encoder(job_dir).encode(out)
    del out
    gc.collect()
    with open(os.path.join(job_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f)


# ----------------------------------------------------------------------------
# Tasks


def versions(store_dir, check=False):
    """Versions of Python and the packages; with check, import the modules used."""
    out = {
        "python": platform.python_version(),
        "executable": sys.executable,
        "machine": platform.machine(),
        "platform": platform.platform(),
    }
    for name in [
        "osl-dynamics",
        "tensorflow",
        "tensorflow-probability",
        "tf-keras",
        "numpy",
    ]:
        try:
            out[name.replace("-", "_")] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name.replace("-", "_")] = ""
    out["cpu_features"] = _cpu_features()
    if check:
        t0 = time.perf_counter()
        import tensorflow  # noqa: F401

        out["tensorflow_import_seconds"] = time.perf_counter() - t0
        from osl_dynamics.data import Data  # noqa: F401
        from osl_dynamics.inference import modes  # noqa: F401
        from osl_dynamics.models import hmm, load  # noqa: F401
        from osl_dynamics.utils.misc import set_random_seed  # noqa: F401
    return out


def train_hmm(
    store_dir,
    data,
    sampling_frequency,
    config,
    model_dir,
    prepare=None,
    seed=None,
    data_field="X",
    time_axis_first=True,
):
    """Prepare the data, train an HMM and infer the state time courses.

    data is a list of [samples x channels] arrays or file paths, one per
    session. The model, the preparation methods and the PCA components are
    saved in model_dir.
    """
    from osl_dynamics.models import hmm
    from osl_dynamics.utils.misc import set_random_seed

    if seed is not None:
        set_random_seed(int(seed))
    prepare = dict(prepare or {})
    offset = _offset(prepare)
    if sum(_method(name) in ("pca", "tde_pca") for name in prepare) > 1:
        raise ValueError("The preparation can have only one PCA step (pca or tde_pca).")

    d = _data(data, sampling_frequency, store_dir, data_field, time_axis_first)
    n_in = [x.shape[0] for x in _sessions(d.time_series())]
    d.prepare(prepare)
    _check_lengths(n_in, d, offset)

    cfg = {
        "sequence_length": 200,
        "batch_size": 256,
        "learning_rate": 0.01,
        "n_epochs": 20,
        "learn_means": False,
        "learn_covariances": True,
    }
    cfg.update(config)
    cfg["n_channels"] = d.n_channels
    for k in [
        "n_states",
        "sequence_length",
        "batch_size",
        "n_epochs",
        "n_init",
        "n_init_epochs",
        "best_of",
    ]:
        if k in cfg:
            cfg[k] = int(cfg[k])
    for k in ["learn_means", "learn_covariances"]:
        cfg[k] = bool(cfg[k])

    model = hmm.Model(hmm.Config(**cfg))
    t0 = time.perf_counter()
    model.train(d)
    train_seconds = time.perf_counter() - t0
    model.save(model_dir)
    with open(os.path.join(model_dir, "osld_prepare.json"), "w", encoding="utf-8") as f:
        json.dump(prepare, f)
    pca = getattr(d, "pca_components", None)
    if pca is not None:
        np.save(os.path.join(model_dir, "osld_pca_components.npy"), pca)

    out = _states(model, d, n_in, offset, sampling_frequency)
    means, covs = model.get_means_covariances()
    out.update(
        free_energy=float(model.free_energy(d)),
        initial_state_probs=np.asarray(model.get_initial_state_probs()),
        trans_prob=np.asarray(model.get_trans_prob()),
        means=np.asarray(means).T,
        covs=np.moveaxis(np.asarray(covs), 0, -1),
        config={k: v for k, v in cfg.items() if isinstance(v, (bool, int, float, str))},
        train_seconds=train_seconds,
    )
    if pca is not None:
        out["pca_components"] = np.asarray(pca)
    return out


def infer(
    store_dir,
    model_dir,
    data,
    sampling_frequency,
    dual_estimation=False,
    n_jobs=1,
    data_field="X",
    time_axis_first=True,
):
    """State time courses of data with a trained model (optionally dual estimation).

    The preparation saved by train_hmm is applied, with the training PCA
    components.
    """
    from osl_dynamics import models

    with open(os.path.join(model_dir, "osld_prepare.json"), encoding="utf-8") as f:
        prepare = json.load(f)
    pca_file = os.path.join(model_dir, "osld_pca_components.npy")
    if os.path.exists(pca_file):
        pca = np.load(pca_file)
        for name, kwargs in prepare.items():
            if _method(name) in ("pca", "tde_pca"):
                kwargs.pop("n_pca_components", None)
                kwargs["pca_components"] = pca
    offset = _offset(prepare)

    model = models.load(model_dir)
    d = _data(data, sampling_frequency, store_dir, data_field, time_axis_first)
    n_in = [x.shape[0] for x in _sessions(d.time_series())]
    d.prepare(prepare)
    _check_lengths(n_in, d, offset)

    out = _states(model, d, n_in, offset, sampling_frequency)
    if dual_estimation:
        t0 = time.perf_counter()
        means, covs = model.dual_estimation(d, n_jobs=int(n_jobs))
        if np.ndim(means) == 2:  # one session
            means, covs = [means], [covs]
        out["dual_estimation_seconds"] = time.perf_counter() - t0
        out["means"] = [np.asarray(m).T for m in means]
        out["covs"] = [np.moveaxis(np.asarray(c), 0, -1) for c in covs]
    return out


def busy(store_dir, seconds=60):
    """Import TensorFlow and run matrix products for a number of seconds."""
    import tensorflow as tf

    x = tf.random.normal([512, 512])
    t0 = last = time.perf_counter()
    n = 0
    while time.perf_counter() - t0 < float(seconds):
        x = tf.linalg.normalize(tf.matmul(x, x))[0]
        n += 1
        if time.perf_counter() - last > 5:
            last = time.perf_counter()
            print(f"busy: {last - t0:.0f} s", flush=True)
    return {"iterations": n, "pid": os.getpid()}


TASKS = {
    "versions": versions,
    "train_hmm": train_hmm,
    "infer": infer,
    "busy": busy,
}


# ----------------------------------------------------------------------------
# Helpers


def _cpu_features():
    """SIMD extensions NumPy detects on this CPU (as seen by this process)."""
    try:
        from numpy._core._multiarray_umath import __cpu_features__
    except ImportError:
        return {}
    return {
        k: bool(__cpu_features__.get(k, False))
        for k in ["SSE42", "AVX", "AVX2", "FMA3", "AVX512F"]
    }


def _sessions(x):
    """A list of 2D arrays, one per session."""
    if isinstance(x, np.ndarray):
        x = [x]
    return [np.asarray(a) for a in x]


def _data(data, sampling_frequency, store_dir, data_field, time_axis_first):
    """An osl-dynamics Data object from arrays or file paths."""
    from osl_dynamics.data import Data

    if isinstance(data, (str, np.ndarray)):
        data = [data]
    inputs = [
        x if isinstance(x, str) else np.ascontiguousarray(x, dtype=np.float32)
        for x in data
    ]
    os.makedirs(store_dir, exist_ok=True)
    return Data(
        inputs,
        data_field=data_field,
        time_axis_first=bool(time_axis_first),
        sampling_frequency=sampling_frequency,
        store_dir=store_dir,
        n_jobs=1,
    )


def _method(name):
    """Data method name of a prepare key ("standardize_2" -> "standardize")."""
    return re.match(r"^(\w+?)(_\d+)?$", name).group(1)


def _offset(prepare):
    """Samples lost at the start of each session by the preparation methods."""
    offset = 0
    for name, kwargs in prepare.items():
        method = _method(name)
        if method in ("tde", "tde_pca"):
            offset += int(kwargs["n_embeddings"]) // 2
        elif method == "moving_average":
            offset += int(kwargs["n_window"]) // 2
        elif method in ("downsample", "filter_and_downsample"):
            raise ValueError(
                f"Preparation method '{method}' changes the sampling rate: downsample the data before."
            )
    return offset


def _check_lengths(n_in, d, offset):
    n_prep = [x.shape[0] for x in _sessions(d.time_series())]
    for i, (a, b) in enumerate(zip(n_in, n_prep)):
        if a - b != 2 * offset:
            raise ValueError(
                f"Session {i + 1}: the preparation keeps {b} of {a} samples, which cannot be aligned with the input."
            )


def _states(model, d, n_in, offset, sampling_frequency):
    """State probabilities, Viterbi paths (1-based) and summary statistics on the input samples."""
    from osl_dynamics.inference import modes

    K = model.config.n_states
    alpha = _sessions(model.get_alpha(d))
    vpath = _sessions(model.get_viterbi_path(d))
    stats = {
        "fractional_occupancy": np.reshape(
            modes.fractional_occupancies(vpath), (len(vpath), K)
        ),
        "mean_lifetime": np.reshape(
            modes.mean_lifetimes(vpath, sampling_frequency), (len(vpath), K)
        ),
        "mean_interval": np.reshape(
            modes.mean_intervals(vpath, sampling_frequency), (len(vpath), K)
        ),
        "switching_rate": np.reshape(
            modes.switching_rates(vpath, sampling_frequency), (len(vpath), K)
        ),
    }
    out = {"alpha": [], "path": [], "stats": stats}
    for a, v, n in zip(alpha, vpath, n_in):
        A = np.full((n, K), np.nan)
        A[offset : offset + a.shape[0]] = a
        P = np.full(n, np.nan)
        P[offset : offset + v.shape[0]] = np.argmax(v, axis=1) + 1
        out["alpha"].append(A)
        out["path"].append(P)
    return out


def _decode(x, job_dir):
    if isinstance(x, dict):
        if "ndarray" in x:
            a = np.fromfile(
                os.path.join(job_dir, x["ndarray"]), dtype=x.get("dtype", "float64")
            )
            return a.reshape([int(n) for n in x["shape"]], order="F")
        return {k: _decode(v, job_dir) for k, v in x.items()}
    if isinstance(x, list):
        return [_decode(v, job_dir) for v in x]
    return x


class _Encoder:
    def __init__(self, job_dir):
        self.job_dir = job_dir
        self.n = 0

    def encode(self, x):
        if isinstance(x, dict):
            return {k: self.encode(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            # wrapped so that MATLAB decodes a list of one item as a cell array
            return {"list": [self.encode(v) for v in x]}
        if isinstance(x, np.ndarray) and x.ndim > 0:
            self.n += 1
            name = f"out{self.n}.bin"
            a = np.asarray(x, dtype=np.float64)
            a.ravel(order="F").tofile(os.path.join(self.job_dir, name))
            return {"ndarray": name, "shape": list(a.shape), "dtype": "float64"}
        if isinstance(x, np.generic):
            x = x.item()
        if isinstance(x, float) and not math.isfinite(x):
            return None
        return x


if __name__ == "__main__":
    try:
        run(sys.argv[1])
    except Exception:
        traceback.print_exc()
        try:
            with open(
                os.path.join(sys.argv[1], "error.txt"), "w", encoding="utf-8"
            ) as f:
                f.write(traceback.format_exc())
        except OSError:
            pass
        sys.exit(1)
