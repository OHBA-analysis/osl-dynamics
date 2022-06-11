:orphan:

Hidden Markov Model (HMM)
=========================

The HMM is a popular model for studying time series data. Our group has previously implemented an HMM in MATLAB: `HMM-MAR <https://github.com/OHBA-analysis/HMM-MAR>`_.
This model has been used successfully  to study brain data [1-7].

This package contains a Python implementation of the HMM. The model in HMM-MAR is fully Bayesian,
whereas the model in this package is only Bayesian in the hidden state inference and learns point-estimates for observation model parameters.

References
----------

#. Vidaurre D, Abeysuriya R, Becker R, Quinn A … Woolrich M. Discovering dynamic brain networks from big data in rest and task. NeuroImage, 2018.
#. Baker AP, Brookes MJ, Rezek IA, Smith SM, Behrens T, Probert Smith PJ, Woolrich M. Fast transient networks in spontaneous human brain activity. Elife 2014.
#. Vidaurre, D., Smith, S. M., Woolrich, M. Brain Network Dynamics are Hierarchically Organised in Time. PNAS 2017. 
#. Vidaurre D, Hunt L, Quinn AJ, Hunt BAE, Brookes MJ, Nobre AC, Woolrich M.  Spontaneous cortical activity transiently organises into frequency specific phase-coupling networks. Nature Communications. 2018.
#. Van Schependom J, Vidaurre D, Costers L, Sjogard M, Sima D, Smeets D, D'Hooghe M, D'Haeseleer M, Deco G, Wens V, De Tiege X, Goldman S, Woolrich M, Nagels G. Reduced brain integrity slows down and increases low alpha power in multiple sclerosis. Multiple Sclerosis Journal, 2020.
#. Sitnikova T, Hughes J, Ahlfors S, Woolrich M, Salat D. Short timescale abnormalities in the states of spontaneous synchrony in the functional neural networks in Alzheimer's disease. NeuroImage: Clinical, 2018.
#. Quinn A, Vidaurre D, Abeysuriya R, Becker R… Woolrich M. Task-evoked dynamic network analysis through hidden markov modelling. Frontiers in Neuroscience, 2018.
#. Higgins C, Liu Y, Vidaurre D, Kurth-Nelson Z, Dolan R, Behrens T, Woolrich M. Replay bursts in humans coincide with activation of the default mode and parietal alpha networks. Neuron, 2021.
