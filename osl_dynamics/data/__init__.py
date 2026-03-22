"""Data loading and preparation.

The main entry point is the :py:class:`Data` class, which handles loading
time series data from various formats (NumPy, MATLAB, FIF) and preparing
it for model training.

Modules
-------
- :py:mod:`~osl_dynamics.data.base` — :py:class:`Data` class for loading
  and preparing data, and :py:class:`SessionLabels` for associating
  metadata with sessions.
- :py:mod:`~osl_dynamics.data.processing` — Data preparation methods
  (standardisation, time-delay embedding, PCA, amplitude envelope) called
  via :py:meth:`Data.prepare`.
- :py:mod:`~osl_dynamics.data.rw` — Low-level read/write functions for
  different file formats.
- :py:mod:`~osl_dynamics.data.task` — Utilities for epoching and working
  with task data.
- :py:mod:`~osl_dynamics.data.tf` — TensorFlow dataset utilities (TFRecord
  I/O, batching).

Tutorials
---------
- :doc:`Loading Data </tutorials_build/1-1_data_loading>`
- :doc:`Preparing M/EEG Data </tutorials_build/1-2_data_prepare_meg>`
- :doc:`Preparing fMRI Data </tutorials_build/1-3_data_prepare_fmri>`
- :doc:`Time-Delay Embedding </tutorials_build/1-4_data_time_delay_embedding>`
"""

from osl_dynamics.data.base import Data, SessionLabels
from osl_dynamics.data.tf import load_tfrecord_dataset

__all__ = ["Data", "SessionLabels", "load_tfrecord_dataset"]
