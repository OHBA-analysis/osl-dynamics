Statistical Significance Testing
--------------------------------

Example scripts for statistical significance testing with `osl_dynamics.stats`:

- `evoked_response.py`: is a network response different from zero? Maximum statistic permutation test.
- `network_response_cluster.py`: the same question with a cluster-based permutation test and TFCE.
- `group_difference.py`: do two groups differ? Row-shuffle permutation test with covariates.
- `three_groups.py`: do three groups differ? F-test for any difference, then pairwise tests.
- `regression.py`: does a quantity change with age? Permutation test of a continuous regressor.

See the [statistics documentation](https://osl-dynamics.readthedocs.io/en/latest/stats.html) for which test and correction to use.
