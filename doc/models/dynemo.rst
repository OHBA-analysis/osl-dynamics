:orphan:

Dynamic Network Modes (DyNeMo)
==============================

There are two flavors of Dynamic Network Modes: DyNeMo and M-DyNeMo.

DyNeMo
------

DyNeMo overcomes two limitations of the HMM:

- The Markovian assumption: DyNeMo uses an LSTM to model long-range temporal dependencies in the latent variable.
- The mutual exclusivity of states: DyNeMo models the data using a linear mixture of `modes`.

DyNeMo was used to study MEG data in [1].

M-DyNeMo
--------

DyNeMo models the data using a single time scale. M-DyNeMo furthers our modelling capability by including multiple dynamics: one for the mean activity and another for the connectivity.

References
----------

#. Chetan Gohil, Evan Roberts, Ryan Timms, Alex Skates, Cameron Higgins, Andrew Quinn, Usama Pervaiz, Joost van Amersfoort, Pascal Notin, Yarin Gal, Stanislaw Adaszewski, and Mark Woolrich, Mixtures of large-scale dynamic functional brain network modes, `bioRxiv preprint <https://www.biorxiv.org/content/10.1101/2022.05.03.490453v1>`_.
