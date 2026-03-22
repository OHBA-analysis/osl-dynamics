"""Data loading and preparation.

The main entry point is the :py:class:`Data` class, which handles loading
time series data from various formats (NumPy, MATLAB, FIF) and preparing
it for model training.

See also
--------
- :doc:`Loading Data </tutorials_build/1-1_data_loading>`
- :doc:`Preparing M/EEG Data </tutorials_build/1-2_data_prepare_meg>`
- :doc:`Preparing fMRI Data </tutorials_build/1-3_data_prepare_fmri>`
- :doc:`Time-Delay Embedding </tutorials_build/1-4_data_time_delay_embedding>`

Modules
-------
- ``base.py`` — :py:class:`Data` class for loading and preparing data, and
  :py:class:`SessionLabels` for associating metadata with sessions.
- ``processing.py`` — Data preparation methods (standardisation, time-delay
  embedding, PCA, amplitude envelope) called via :py:meth:`Data.prepare`.
- ``rw.py`` — Low-level read/write functions for different file formats.
- ``task.py`` — Utilities for epoching and working with task data.
- ``tf.py`` — TensorFlow dataset utilities (TFRecord I/O, batching).
"""

from osl_dynamics.data.base import Data, SessionLabels
from osl_dynamics.data.tf import load_tfrecord_dataset

__all__ = ["Data", "SessionLabels", "load_tfrecord_dataset"]
