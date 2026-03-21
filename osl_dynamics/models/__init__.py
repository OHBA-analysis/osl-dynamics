"""Generative models.

This subpackage contains all the models implemented in osl-dynamics. Each
model module (e.g. ``hmm.py``, ``dynemo.py``) defines a ``Config`` dataclass
and a ``Model`` class.

Code structure
--------------

The code is organised into three layers:

**1. Base layer** (``mod_base.py``)

- :py:class:`~osl_dynamics.models.mod_base.BaseModelConfig` — Common
  configuration shared by all models (learning rate, batch size, number of
  modes/states, etc.).
- :py:class:`~osl_dynamics.models.mod_base.ModelBase` — Abstract base class
  that wraps a Keras model. Provides the training loop (``fit``),
  initialisation, checkpointing, and attribute delegation to the underlying
  Keras model. Subclasses must implement ``build_model()``.

**2. Inference layer** (``inf_mod_base.py``)

Two parallel branches extend ``ModelBase`` for different inference paradigms:

- **Variational inference** — For models with continuous latent variables
  (mode mixing coefficients inferred by an RNN). Adds KL annealing and
  alpha temperature handling.

  - :py:class:`~osl_dynamics.models.inf_mod_base.VariationalInferenceModelConfig`
  - :py:class:`~osl_dynamics.models.inf_mod_base.VariationalInferenceModelBase`
  - Used by: DyNeMo, M-DyNeMo, SC-DyNeMo, DIVE, DyNeStE.

- **Markov state inference** — For models with discrete hidden states
  (state sequence inferred by the Baum-Welch algorithm). Adds transition
  probability learning and state initialisation.

  - :py:class:`~osl_dynamics.models.inf_mod_base.MarkovStateInferenceModelConfig`
  - :py:class:`~osl_dynamics.models.inf_mod_base.MarkovStateInferenceModelBase`
  - Used by: HMM, HMM-Poisson, HIVE.

**3. Full model**

Each model combines a ``Config`` (via multiple inheritance from
``BaseModelConfig`` + an inference config) and a ``Model`` (inheriting from
the appropriate inference base class):

.. list-table::
   :header-rows: 1
   :widths: 20 15 50

   * - Model
     - Inference
     - Description
   * - :py:mod:`~osl_dynamics.models.hmm`
     - Markov
     - Hidden Markov Model with MVN observations.
       See :doc:`model description </models/hmm>`.
   * - :py:mod:`~osl_dynamics.models.hmm_poi`
     - Markov
     - HMM with Poisson observations.
   * - :py:mod:`~osl_dynamics.models.hive`
     - Markov
     - HMM with Integrated Variability Estimation
       (session-specific parameters via embeddings).
       See :doc:`model description </models/hive>`.
   * - :py:mod:`~osl_dynamics.models.dynemo`
     - Variational
     - Dynamic Network Modes (continuous mode mixing via RNN).
       See :doc:`model description </models/dynemo>`.
   * - :py:mod:`~osl_dynamics.models.mdynemo`
     - Variational
     - Multi-Dynamic Network Modes (separate dynamics for
       power and connectivity).
       See :doc:`model description </models/mdynemo>`.
   * - :py:mod:`~osl_dynamics.models.sc_dynemo`
     - Variational
     - Single-Channel DyNeMo (extends DyNeMo).
   * - :py:mod:`~osl_dynamics.models.dive`
     - Variational
     - DyNeMo with Integrated Variability Estimation.
   * - :py:mod:`~osl_dynamics.models.dyneste`
     - Variational
     - Dynamic Network States (discrete states with
       non-Markovian temporal model).
       See :doc:`model description </models/dyneste>`.

**Utilities** (``obs_mod.py``)

Shared functions for getting/setting observation model parameters
(means, covariances, embeddings, regularizers).
"""

import yaml

from osl_dynamics.models import (
    dynemo,
    mdynemo,
    sc_dynemo,
    hmm,
    hmm_poi,
    hive,
    dive,
    dyneste,
)
from osl_dynamics.utils import misc

models = {
    "DyNeMo": dynemo.Model,
    "M-DyNeMo": mdynemo.Model,
    "SC-DyNeMo": sc_dynemo.Model,
    "HMM": hmm.Model,
    "HMM-Poisson": hmm_poi.Model,
    "HIVE": hive.Model,
    "DIVE": dive.Model,
    "DyNeStE": dyneste.Model,
}


def load(dirname, single_gpu=True):
    """Load model.

    Parameters
    ----------
    dirname : str
        Path to directory where the config.yml and weights are stored.
    single_gpu : bool, optional
        Should we compile the model on a single GPU?

    Returns
    -------
    model : osl-dynamics model
        Model object.
    """
    with open(f"{dirname}/config.yml", "r") as file:
        config_dict = yaml.load(file, misc.NumpyLoader)

    if "model_name" not in config_dict:
        raise ValueError(
            "Either use a specific `Model.load` method or "
            "provide a `model_name` field in config"
        )

    try:
        model_type = models[config_dict["model_name"]]
    except KeyError:
        raise NotImplementedError(
            f"{config_dict['model_name']} was not found. "
            f"Options are {', '.join(models.keys())}"
        )

    return model_type.load(dirname, single_gpu=single_gpu)
