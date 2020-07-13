.. include:: <isonum.txt>

=========
Changelog
=========

Version 1.1.0
=============

- API Changes
    - variational_rnn_autoencoder
        - burnin_epochs |rarr| n_epochs_burnin
        - initial_pseudo_cov |rarr| initial_covariances (can accept non-cholesky matrices)
        - learn_covs |rarr| learn_covariances
        - activation_function |rarr| alpha_xform
        - Added n_layers_inference, n_layers_model.
