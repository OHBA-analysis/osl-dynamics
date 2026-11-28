"""Train a small HMM on simulated data and print timings (MATLAB interface CI)."""

import itertools
import platform
import time

import numpy as np

try:
    from numpy._core._multiarray_umath import __cpu_features__ as cpu
except ImportError:
    cpu = {}

print("Python", platform.python_version(), platform.machine(), platform.platform())
print("CPU features:", {k: bool(cpu.get(k)) for k in ["SSE42", "AVX", "AVX2", "FMA3", "AVX512F"]})

t0 = time.perf_counter()
import tensorflow as tf  # noqa: E402

print(f"TensorFlow {tf.__version__} imported in {time.perf_counter() - t0:.1f} s")

from osl_dynamics.data import Data  # noqa: E402
from osl_dynamics.models import hmm  # noqa: E402
from osl_dynamics.utils.misc import set_random_seed  # noqa: E402

# Zero-mean Gaussian states with different covariances and a Markov chain
rng = np.random.default_rng(1)
K, C, n = 3, 8, 20000
A = 0.98 * np.eye(K) + 0.02 / (K - 1) * (1 - np.eye(K))
R = []
for k in range(K):
    Q, _ = np.linalg.qr(rng.standard_normal((C, C)))
    R.append(np.linalg.cholesky(Q @ np.diag(np.linspace(0.2, 3, C) ** (k / 2 + 0.5)) @ Q.T + 0.05 * np.eye(C)))
z = np.zeros(n, dtype=int)
for t in range(1, n):
    z[t] = rng.choice(K, p=A[z[t - 1]])
x = np.zeros((n, C), dtype=np.float32)
for k in range(K):
    x[z == k] = rng.standard_normal((np.sum(z == k), C)) @ R[k].T

set_random_seed(1)
data = Data(x, sampling_frequency=250)
data.standardize()
config = hmm.Config(
    n_states=K,
    n_channels=C,
    sequence_length=200,
    learn_means=False,
    learn_covariances=True,
    batch_size=16,
    learning_rate=0.01,
    n_epochs=10,
    n_init=3,
    n_init_epochs=1,
)
model = hmm.Model(config)
t0 = time.perf_counter()
model.train(data)
seconds = time.perf_counter() - t0
path = np.argmax(model.get_viterbi_path(data), axis=1)
acc = max(np.mean(np.array(p)[path] == z[: len(path)]) for p in itertools.permutations(range(K)))
print(f"Trained in {seconds:.1f} s ({seconds / 13:.2f} s per epoch); accuracy {acc:.3f}")
data.delete_dir()
assert acc > 0.9, "The HMM did not recover the simulated states."
