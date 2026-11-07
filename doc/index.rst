:html_theme.sidebar_secondary.remove:

OSL Dynamics
============

.. rst-class:: lead

   A Python toolbox for studying dynamic functional brain networks in M/EEG and fMRI data: from raw recordings to parcel time courses, generative models of network dynamics, and statistical significance testing.

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: :octicon:`download` Install
      :link: install
      :link-type: doc

      Install from conda-forge or PyPI, and check your GPU is working.

   .. grid-item-card:: :octicon:`rocket` Getting Started
      :link: getting_started
      :link-type: doc

      A quick introduction: a typical workflow and a short example.

   .. grid-item-card:: :octicon:`mortar-board` Tutorials
      :link: tutorials
      :link-type: doc

      Each stage of an analysis on real data, as scripts and notebooks you can download.

   .. grid-item-card:: :octicon:`pulse` M/EEG Processing
      :link: meeg
      :link-type: doc

      Preprocessing, coregistration, source reconstruction, parcellation and sign flipping.

   .. grid-item-card:: :octicon:`git-branch` Models
      :link: models/index
      :link-type: doc

      The HMM, DyNeMo and other generative models for dynamic brain networks.

   .. grid-item-card:: :octicon:`graph` Statistics
      :link: stats
      :link-type: doc

      GLM permutation tests with maximum statistic, cluster-based and TFCE corrections.

   .. grid-item-card:: :octicon:`apps` Parcellations
      :link: parcellations/index
      :link-type: doc

      The brain parcellations available in osl-dynamics.

   .. grid-item-card:: :octicon:`question` FAQ
      :link: faq
      :link-type: doc

      Common questions on preparing data, training models and post-hoc analysis.

   .. grid-item-card:: :octicon:`code` API Reference
      :link: autoapi/index
      :link-type: doc

      Every class and function, with its parameters.

Citing osl-dynamics
-------------------

If you find this toolbox useful, please cite:

   Gohil, C., Huang, R., Roberts, E., van Es, M. W., Quinn, A. J., Vidaurre, D., & Woolrich, M. W. (2024). `osl-dynamics, a toolbox for modeling fast dynamic brain activity <https://elifesciences.org/articles/91949>`_. *eLife*, 12, RP91949.

Getting help
------------

To request a new feature or report a bug, please create an issue on `GitHub <https://github.com/OHBA-analysis/osl-dynamics/issues>`_.

osl-dynamics is developed by the `OHBA Methods Group <https://www.psych.ox.ac.uk/research/ohba-analysis-group>`_ at the University of Oxford.

.. raw:: html

   <div class="institution-logos">
     <a href="https://www.oxcin.ox.ac.uk/about/our-locations/OHBA"><img src="https://raw.githubusercontent.com/OHBA-analysis/osl-files/main/docs/logo/ohba-logo.png" alt="OHBA"></a>
     <a href="https://www.oxcin.ox.ac.uk"><img src="https://raw.githubusercontent.com/OHBA-analysis/osl-files/main/docs/logo/oxcin-logo.png" alt="Oxford Centre for Integrative Neuroimaging"></a>
   </div>

.. toctree::
   :hidden:

   Install <install>
   Getting Started <getting_started>
   User Guide <documentation>
   Tutorials <tutorials>
   FAQ <faq>
