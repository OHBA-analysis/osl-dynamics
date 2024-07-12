Frequently Asked Questions (FAQ)
================================

.. contents::
   :local:

If you  have a question that's not listed above, please email chetan.gohil@psych.ox.ac.uk and we may add it to this page.

Installation
------------

How do I install osl-dynamics?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The recommended installation of the latest version via pip is described :doc:`here <install>`.

Do I need a GPU?
~~~~~~~~~~~~~~~~

No, osl-dynamics can be used on solely on CPUs, however, using the package on a computer with a GPU will be much faster (could expect a speed up of 10x). Note, `Google colab <https://colab.research.google.com/>`_ offers free GPU access.

The GPU use in this package is via the TensorFlow package, which is used by osl-dynamics to create and train the models (HMM, DyNeMo, etc).

How do I use a GPU?
~~~~~~~~~~~~~~~~~~~

GPU use in this package is via the `TensorFlow <https://www.tensorflow.org>`_ package. GPUs can only be used with GPU-enabled installations of TensorFlow. Whether you're able to install a GPU-enabled version of TensorFlow depends on your computer (hardware and operating system). The TensorFlow installation `webpage <https://www.tensorflow.org/install/pip>`_ describes how to install TensorFlow with GPU support for various operating systems. If you're able to install a GPU-enabled version of TensorFlow via the instructions on this website, you will automatically use any GPUs you have.

Note:

- There's no official TensorFlow package in pip which supports GPU use on a MacOS, however, you can install tensorflow-metal following the instructions `here <https://developer.apple.com/metal/tensorflow-plugin/>`_, which will allow TensorFlow to take advantage of hardware inside your Mac.

- If you're installing osl-dynamics on a computing cluster you know has GPUs, then you may only need to ensure you have the correct CUDA/cuDNN libraries installed/loaded and you will automatically use the available GPUs. You could also consider installing a GPU-enabled version of TensorFlow called :code:`tensorflow-gpu` via conda or pip.

Data
----

How do I optimally preprocess my data for training an HMM or DyNeMo?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For electrophysiological data we have found preprocessing the sensor-level data by downsampling to 250 Hz and bandpass filtering 1-45 Hz works well. Additionally, we do some bad segment detection based on the variance of non-overlapping windows. Following this, we use a volumetric linearly constrained minimum variance (LCMV) beamformer to estimate source space data. Usually the beamformed (voxel) data is parcellated to ~40 regions of interest and an orthogonalisation technique is used to correct for spatial leakage. Additionally, the sign of parcel time courses is adjusted to align across subjects. All of these steps can be done using the `OSL package <https://github.com/OHBA-analysis/osl>`_ in Python.

The 2023 OSL workshop had a session on dynamic network modelling. The OSF project hosting workshop materials (`here <https://osf.io/zxb6c/>`_) has a jupyter notebook with recommended preprocessing and source reconstruction for fitting the HMM/DyNeMo. See the **Dynamics/practicals/0_preproc_source_recon_and_sign_flip.ipynb** tutorial.

Can I use a structural parcellation (e.g. AAL or Desikan-Killiany atlas)?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes, you can. There’s nothing stopping you from using the parcellation you want during source reconstruction. Note, source reconstruction is performed using the `OSL package <https://github.com/OHBA-analysis/osl>`_ rather than within osl-dynamics.

However, you want the number of parcels to be a good amount less than the rank of the sensor space data in order to well estimate your parcel time courses. The rank is at most equal to the number of sensors you have. With Maxfiltered data (e.g. Elekta/MEGIN data), the default rank of the sensor data is ~64, and so it is sensible to require the number of parcels to be less than 64. Even with non-maxfiltered data with hundreds of sensors (e.g. CTF, OPMs) the effective amount of information in the sensor data typically corresponds to a rank of about 100. You can look at the eigenspectrum of your sensor space data to check this.

Also note, the requirement to have the number of parcels less than the rank is an absolute requirement, if you are using the recommended **symmetric orthogonalisation** approach on the parcel time courses to correct for spatial leakage. This is not a deficiency of the symmetric orthogonalisation approach, but reflects the rank needed to use this more complete spatial leakage correction (it removes so-called inherited or ghost interactions as well) while still being able to estimate parcel time courses unambiguously.

Why do I need to sign flip my data before training a model?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A discussion of why dipole sign flipping is needed is covered in the **Dynamics/practicals/0_sign_fliping.ipynb** tutorial in the 2023 OSL workshop: `here <https://osf.io/mjn8r>`_.

In short, you can calculate a covariance matrix using the time series data from each subject. The sign of the off-diagonal elements in the covariance matrix may not be the same across subjects. I.e. channels i and j maybe positively correlated for one subject but negatively correlated for another. The subjects can be aligned by flipping the sign of channel i or j for one of the subjects - this is the 'sign flipping'. This is important because the HMM/DyNeMo models dynamic changes in the covariance of the data, we do not want dynamics in the covariance simply due to misaligned signs.

Note, the sign ambiguity is an identifiability problem in the source reconstruction step that cannot be avoided.

Do I need to prepare my data before training a model?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No, strictly speaking you don't need to prepare your data before training a model. However, you are much more likely to infer a reasonable description of your data if you follow a pipeline that has previously been shown to work. **Therefore, it is recommended that you prepare the data**.

There are three common choices for preparing the data:

#. **Just standardize**. Here, all we do is z-transform the data (subtract the mean and divide by the standard deviation for each channel individually). Standardization is helpful for the optimization process used to train the models in osl-dynamics. **This is the recommended approach for studying sensor-level M/EEG data or fMRI data**.

#. **Calculate amplitude envelope data and standardize**. This is common approach for overcoming the dipole sign ambiguity problem in MEG - where the sign of source reconstructed channels can be misaligned cross different subjects or sessions. Here, it is common to apply a Hilbert transform to the 'raw' data and apply a short sliding window to smooth the data. The amplitude envelope can be thought of as an analogous effect to the hemodynamic response blurring out neuronal signals in fMRI.

#. **Calculate time-delay embedded data, followed by principal component analysis and standardization**. Time-delay embedding is described in the 'What is time-delay embedding?' section below. **This is the recommended approach for studying source-space M/EEG data**.

The :doc:`Preparing Data tutorial <tutorials_build/data_prepare_meg>` covers how to prepare data using the three options.

What is time-delay embedding?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Time-Delay Embedding (TDE) involves adding extra channels containing time-lagged versions of the original data:

- For each channel you shift the time series by a fixed amount (forwards or backwards) and add it as an extra channel to the time series data.
- You do this for a pre-specified number of lags. **Typically, we use lags of -7,-6,...,6,7** (equivalent to :code:`n_embeddings=15`). This results in an extra 14 channels being added for each original channel. E.g. if you originally had 10 channels and added ±7 lags, you would end up with a time series with 150 channels.

The purpose of TDE is to encode spectral (frequency-specific) information in the covariance matrix of the data. The covariance matrix of the TDE data has additional off-diagonal elements which corresponds to the auto-correlation function (this characterises the spectral properties of the original data). TDE is useful when we want to model transient spectral properties in the data.

Usually adding the extra channels results in a very high-dimensional time series, so we typically also apply principal component analysis for dimensionality reduction. **We recommend reducing down to at least twice the number of original channels**.

How does the choice of number of lags (embeddings) affect the model when doing time-delay embedding?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See the 'What is time-delay embedding' question for a description of what happens when we perform time-delay embedding (TDE).

A choice we have to make is how many lagged version of each channel we add. The number of lagged channels we add (i.e. the number of embeddings) determines how many points in the auto-correlation function (and therefore power spectrum) we encode into the covariance matrix of the data. I.e. if we include more embeddings, we add more off-diagonal elements into the covariance matrix, which corresponds to specifying more data points in the auto-correlation function and therefore power spectrum. In other words, having more embeddings allows you to pick up on smaller differences in the frequency of oscillations in your data - the resolution of the power spectrum has increased.

However, we find with electrophysiological data that there aren't very many narrowband peaks in the power spectrum. Therefore, having a very high resolution power spectrum doesn't affect the state/mode decomposition with the HMM/DyNeMo - rather the HMM/DyNeMo states/modes are more driven by differences in total power instead of frequency specific power (although the frequency content does have some effect, it's not the main driver). Consequently, you will likely find the states/modes you infer with the HMM/DyNeMo aren't very sensitive to the number of embeddings.

The number of embeddings should be chosen for a particular sampling frequency. See the :doc:`Time-Delay Embedding tutorial <tutorials_build/data_time_delay_embedding>` for example code comparing different TDE settings.

Why doesn't the number of time points in my inferred alphas match the original data?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The process of preparing the data before training a model can lead to the loss a data points at the start and end of the time series. This occurs when we perform:

- Time-delay embedding. Here, we lose :code:`n_embeddings // 2` data points from each end of the time series because we don't have the necessary lagged data points before and after the time series to specify the value for each channel.
- Smoothing after a Hilbert transform. When we prepare amplitude envelope data, we usually apply a smoothing window. The length of the window is specified using the :code:`n_window`. When we smooth the data with the window we lose :code:`n_window // 2` data points from each end of the time series.

Note, we have a separate time series for each subject, so we lose these data points from each subject separately. In addition to the data point lost above, before we train a model we separate the time series into sequences. We lose the data points **at the end** that do not form a complete sequence.

Note, you can trim data using the :code:`Data.trim_time_series` method, example use::

    from osl_dynamics.data import Data

    Data = data(...)
    data = data.trim_time_series(n_embeddings=..., sequence_length=...)

Modelling
---------

What is the Hidden Markov Model (HMM)?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See the model description page :doc:`here <models/hmm>`.

What is Dynamic Network Modes (DyNeMo)?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See the model description page :doc:`here <models/dynemo>`.

What model should I use?
~~~~~~~~~~~~~~~~~~~~~~~~

Unfortunately, there is no clear cut answer to this question. The two main models in this package are Dynamic Network Modes (DyNeMo) and the Hidden Markov Model (HMM). Both are valid options. The pros and cons of each are:

- DyNeMo describes the data as a linear mixture of networks, whereas the HMM is a mutually exclusive network model. The lack of mutual exclusivity can actually complicate how we can interpret the data. For resting-state data the access to interpretable summary statistics such as state fractional occupancies, lifetimes, etc. might be worth the comprimised description of the data using mutually exclusive states.
- With task data, the mutual exclusivity can harm the evoked network response and DyNeMo may provide a cleaner description of how the brain responds to a task. See the 2023 OSL workshop tutorial using DyNeMo to study the Wakeman-Henson dataset (`here <https://osf.io/zb7c5>`_) for an illustration of this.
- Practically, without a GPU DyNeMo can be slow to train, whereas the HMM is much quicker.

What is the difference between training on amplitude envelope (AE) and time-delay embedded (TDE) data?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both calculating the AE and TDE are referred to as 'data preparation' options. See the 'Do I need to prepare my data before training a model?' question for a description of how AE and TDE data is calculated.

The models in osl-dynamics (HMM and DyNeMo) aim to describe dynamics in the first and second order statistics of the data, i.e. the mean vector and covariance matrix respectively. We calculate AE or TDE data to ensure the mean and covariance of the data contains dynamics we're interested in.

For example, if we are interested in modelling transient events of high amplitude, we can calculate the AE of our original data and fit an HMM learning the mean vector for multiple states. This will help the HMM find states that have differences in the mean amplitude.

If we are interested in modelling transient bursts of oscillations (spectral events), we can train on TDE data. Each oscillatory burst will have a unique covariance matrix (oscillations at different frequencies will affect the value of off-diagonal elements in the covariance of TDE data). This will help the HMM find states that have different oscillatory behaviour.

Note, when we train on TDE data, because the differences we want to model are reflect in the covariance of the data, we don't need to model the state means (we can just fix them to zero). Whereas, when we train on AE data, the differences we want to model are contained in the mean, so we learn the state means.

Also note, once we have inferred a latent description, such as HMM states or DyNeMo modes, we can go back to the original unprepared data (i.e. before the AE/TDE) and re-estimate properties of this time series based on the inferred latent description. This is what's done when we estimate post-hoc spectra - see the 'Spectral Analysis' section in the model descriptions: :doc:`HMM <models/hmm>` and :doc:`DyNeMo <models/dynemo>`.

What are hyperparameters?
~~~~~~~~~~~~~~~~~~~~~~~~~

There are two types of parameters in osl-dynamics models:

- **Model parameters**. These are parameters that are part of the generative model. These are learnt from the data. E.g. for the HMM, this is the state time course, state means and covariances.
- **Hyperparameters**. Theese are pre-specified parameters that are not learnt from the data. These are the parameters specified in the :code:`Config` object used to create a model. E.g. the number of states/modes, sequence length, batch size, etc are hyperparameters.

How do I choose what hyperparameters to use?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unfortunately, many modern machine learning models come with hyperparameters (parameters that are not part of the generative model) which need to be pre-specified. The best approach is to try and few combinations and do the following:

- Make sure any conclusions are robust to the choice of hyperparameters.
- Use the variational free energy (see the model desciptions in the :doc:`docs <documentation>`) to compare models. Preferably, the variational free energy would be calculated on a hold out validation dataset, which is not used for training.

The `config API <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/config_api/index.html>`_ has two wrapper functions for training an `HMM <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/config_api/wrappers/index.html#osl_dynamics.config_api.wrappers.train_hmm>`_ or `DyNeMo <https://osl-dynamics.readthedocs.io/en/latest/autoapi/osl_dynamics/config_api/wrappers/index.html#osl_dynamics.config_api.wrappers.train_dynemo>`_, which pre-specify hyperparameters that was worked well in the past. These might be a good place to start.

I have trained a model with the same hyperparameters on the same data multiple times, why do I get a different value for the final cost function (variational free energy)?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Modern machine learning models have a problem of **local optima**. When we train a model on complex and noisy data there may be multiple choices for the model parameters (see the 'What are hyperparameters?' question for the definition of 'model parameters') that can lead to similar values for the **cost function**. In our case, the cost function is the **variational free energy** (see the :doc:`HMM description <models/hmm>` for further details). Additionally, different final values for the cost function can occur due to different initial values for the model parameters and the stochasticity in updating the model parameters during training.

Unfortunately, there is no solution to this. With more data this becomes less of a problem. The recommendation is to train a model multiple times and select the model with the best variational free energy for further analysis. Preferably the variational free energy would be calculated using a hold out validation dataset rather than the training data. However, it is common just to compare the variational free energy on the training dataset. Additionally, the recommendation is to make sure any claims are reproducible across a set of runs. Even though you may have runs with different final free energies, in a lot of cases you find the general description (inferred state/mode time courses) give the same picture across runs.

How do I select the optimum number of HMM states or DyNeMo modes?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unfortunately, the number of states/modes in the HMM/DyNeMo needs to be pre-specified. Theoretically, when we do Bayesian inference we can use the **model evidence** to compare models (this include models that differ in terms of the number of states/modes). However, we find with electrophysiological data that the model evidence increases indefinitely with the number os states/modes (tested up to 30 states/modes). I.e. the model evidence is telling us the optimum number of states/modes is above 30, more states/modes is better. However, using a very high number of states defeats the purpose of obtaining a low-dimensional and interpretable description of the data. **Therefore, we suggest using between 6-14 states/modes. 8 states/modes might be a good initial choice**. We find although this number of states/modes might not give the best description of the data from a Bayesian point of view, it still provides a useful description of the data.

Also see the 'How do I choose what hyperparameters to use?' question.

Is there a recommended default initialization for each model?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As explained in 'I have trained a model with the same hyperparameters on the same data multiple times, why do I get a different value for the final cost function (variational free energy)?', the final value for the cost function (model fit) can be sensitive to the initialization of your model parameters. To help with this model have methods to find good initial values for model parameters before doing the full training. The recommended initialization is different for different models. The recommendations are:

- For the HMM use::

    model.random_state_time_course_initialization(n_init=3, n_epochs=1)

- For DyNeMo use::

    model.random_subset_initialization(n_init=5, n_epochs=1, take=0.25)

Why do we use shorter sequences with DyNeMo compared to the HMM?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In most examples you will see a :code:`sequence_length` hyperparameter which is around 200 for DyNeMo and 2000 for the HMM. The sequence is much shorter for DyNeMo because of the components used in the model. DyNeMo uses recurrent neural networks (RNNs), which operate on the data sequentially. This means using very long sequences significantly slows down training the model - it's more effective to separate into shorter sequences that are processed in parallel on a GPU. Note, also fitting longer sequences into GPU memory is difficult when working with RNNs. The HMM on the other hand does not include any RNN components, this means we're able to operate on much longer sequences more efficiently.

Can I do static analysis with this package?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes! See the tutorials :doc:`here <documentation>`.

Can I use this package with task data?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes! the models contained in osl-dynamics can be applied to task data. **The recommended approch is to preprocess/prepare task data as if it is resting-state data and fit an HMM/DyNeMo to it as normal**. When you have inferred a state/mode time course you can then do post-hoc analysis using the task timings, e.g. by epoching the state/mode time course around an event.

Can I train an HMM/DyNeMo on EEG data?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes! You can train the HMM/DyNeMo on sensor-level or source reconstructed EEG data. Note, the same preparation steps (see 'Do I need to prepare my data before training a model?') should be applied to the data irrespective of if it is MEG or EEG data.

I've encountered a NaN when I tried to train a model? Why did this happen and how can I fix it?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Models in osl-dynamics are trained using 'stochastic gradient decent'. We believe the NaN values in the loss function arise from a bad update to the model parameters. We recommend using a lower learning rate and/or larger batch size to help avoid this problem. Alternatively, more aggressive bad segment removal when proprocessing the data seems to help with this problem.

Why is my loss function increasing when I train a DyNeMo model?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The cost function we use for DyNeMo is the variational free energy:

.. math::
    \mathcal{F} = -LL + \eta KL

where :math:`\eta` is the 'KL annealing factor'. This is a scalar that starts off at zero and increases to one as training progresses. At the start of training we suppress the KL term of the loss function (by setting :math:`\eta` to zero), which allows us to find the model that maximises the likliehood (via minimising the negative log-likelihood term, :math:`-LL`). Then we slowly turn on the KL term by increasing :math:`\eta`, this adds more of the KL term to the cost funciton which gives the apparent rise in the loss function. This process is known as **KL annealing**. We typically use KL annealing for the first half of training (set using the :code:`n_kl_annealing_epochs` hyperparameter). After :code:`n_kl_annealing_epochs` of training have occurred then the model is using the full cost function (with :math:`\eta=1`). If we are using early stopping we should make sure we're only considering epochs after :code:`n_kl_annealing_epochs`.

Post-Hoc Analysis
-----------------

What are the most important summary statistics to use?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is common to look at four summary statistics:

- The **fractional occupancy**, which is the fraction of total that is spent in a particular state.
- The **mean lifetime**, which is the average duration of a state visit. This is called known as the 'dwell time'.
- The **mean interval**, which is the average duration between successive state visits.
- The **switching rate**, which is the average number of visits to a state per second.

Summary statistics can be calculated for individual subjects or for a group. See the :doc:`HMM Summary Statistics tutorial <tutorials_build/hmm_summary_stats>` for example code of how to calculate these quantities.

Often, we are interested in comparing two groups or conditions. E.g. we might find static alpha (8-12 Hz) power is increased for one group/condition. Let's speculate there are segments in our data where alpha power bursts occur - this would be identified by the HMM as a state with high alpha power that only activates for particular segments. The increase in alpha power seen for a group/condition can arise in many ways, maybe the alpha bursts are longer in duration, maybe they're more frequency, maybe the dynamics are unchanged but the alpha state just has more alpha power in it. The different summary statistics can potentially help interpret which of these options it is.

Generally, it's difficult to say whether or not one summary statistics is more important than another. The recommended approach is to calculate all four of the above and use the combination of all them as a summary of dynamics for each subject/group/condition.

How does the multitaper spectrum calculation work?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See the 'Spectral Analysis' section of the :doc:`HMM description <models/hmm>`.

How does the regression spectrum calculation work?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See the 'Spectral Analysis' section of the :doc:`DyNeMo model description <models/dynemo>`.

Other
-----

I've used the Matlab HMM-MAR toolbox before, how do I switch to Python?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you're completely new to Python, you may find reading up on how to install Python using Anaconda useful.

If you're familiar with Python and would just like to switch to osl-dynamics to train an HMM and for post-hoc analysis, all you need from the Matlab code is the training data you used in HMM-MAR. It is common in HMM-MAR to save the training data as vanilla :code:`.mat` files using something like the following::

    mat_files = cell(length(subjects_to_do),1);
    T_all = cell(length(subjects_to_do),1);
    for ss = 1:length(subjects_to_do)
        mat_files{ss} = [matrixfilesdir 'subject' num2str(ss) '.mat'];
        [~,T_ss] = read_spm_file(parcellated_Ds{ss},mat_files{ss});
        T_all{ss} = T_ss;
    end

The above code snippet was taken from the example `here <https://github.com/OHBA-analysis/HMM-MAR/blob/master/examples/NatComms2018_fullpipeline.m>`_. If you have the :code:`subject1.mat, subject2.mat, ...` files, you can easily load them into osl-dynamics using the Data class::

    from osl_dynamics.data import Data

    data = Data(['subject1.mat', 'subject2.mat', ...])

And use osl-dynamics as normal.

I found a bug, what do I do?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create an issue `here <https://github.com/OHBA-analysis/osl-dynamics/issues>`_ or email chetan.gohil@psych.ox.ac.uk.

How can I cite the package?
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Please cite the `eLife <https://elifesciences.org/articles/91949>`_ publication:

    **Chetan Gohil, Rukuang Huang, Evan Roberts, Mats WJ van Es, Andrew J Quinn, Diego Vidaurre, Mark W Woolrich (2024) osl-dynamics, a toolbox for modeling fast dynamic brain activity eLife 12:RP91949.**
