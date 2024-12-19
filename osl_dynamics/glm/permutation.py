import logging

import numpy as np
from scipy import stats
from pqdm.processes import pqdm
from tqdm.auto import trange

from osl_dynamics.glm.base import GLM
from osl_dynamics.glm.ols import osl_fit


_logger = logging.getLogger("osl-dynamics")


class Permutation:
    """
    Base class for permutation tests.

    Parameters
    ----------
    design : osl_dynamics.glm.base.Design
        Design object.
    contrast_indx : int
        Index of the contrast of interest.
    n_perm : int
        Number of permutations.
    perm_type : str, optional
        Type of permutation. Options are 'sign_flip' and 'row_shuffle'.
        If None, it will be determined based on the feature types and contrast type.
    n_jobs : int, optional
        Number of jobs to run in parallel. Default is 1.
    """

    def __init__(self, design, contrast_indx, n_perm, perm_type=None, n_jobs=1):
        self.glm = GLM(design)
        self.contrast_indx = contrast_indx
        self.c = self.glm.c[self.contrast_indx][None, :]
        self.n_perm = n_perm
        self.n_jobs = n_jobs
        self.perm_type = self._validate_perm_type(perm_type)

    def permute_X(self):
        """
        Permute the design matrix based on the perm_type.

        Returns
        -------
        X_copy : np.ndarray
            Permuted design matrix. Shape is (n_samples, n_features).
        """
        X_copy = self.glm.X.copy()
        permute_indx = self._get_permute_feature_indx()
        if self.perm_type == "sign_flip":
            # Randomly flip the sign of the features
            signs = np.random.choice([-1, 1], self.glm.n_samples)
            X_copy[:, permute_indx] *= signs[:, None]
        else:
            # Randomly shuffle the rows
            row_indx = np.random.permutation(self.glm.n_samples)
            X_copy[:, permute_indx] = X_copy[np.ix_(row_indx, permute_indx)]

        return X_copy

    def fit(self, y):
        """
        Fit the GLM with unpermuted data and run permutations.

        Parameters
        ----------
        y : np.ndarray
            Target variable. Shape is (n_samples, *target_dims).
        """
        self.glm.fit(y)
        y_flatten = np.reshape(y, (self.glm.n_samples, -1))
        # Build keyword arguments for parallel processing
        kwargs = []
        for _ in range(self.n_perm):
            kwargs.append(
                {
                    "X": self.permute_X(),
                    "y": y_flatten,
                    "contrasts": self.c,
                }
            )

        # Run permutations
        if len(kwargs) == 1:
            _logger.info(
                f"Running permutations on contrast {self.glm.contrast_names[self.contrast_indx]}."
            )
            results = [osl_fit(**kwargs[0])]
        elif self.n_jobs == 1:
            _logger.info(
                f"Running on contrast {self.glm.contrast_names[self.contrast_indx]} permutations with {self.n_jobs} jobs."
            )
            results = []
            for i in trange(self.n_perm, desc="Running permutations"):
                results.append(osl_fit(**kwargs[i]))
        else:
            _logger.info(
                f"Running on contrast {self.glm.contrast_names[self.contrast_indx]} permutations with {self.n_jobs} jobs."
            )
            results = pqdm(
                kwargs,
                osl_fit,
                argument_type="kwargs",
                n_jobs=self.n_jobs,
                desc="Running permutations",
            )

        # Unpack results
        null_copes, null_tstats = [], []
        for result in results:
            _, copes, varcopes = result
            null_copes.append(copes)
            null_tstats.append(self.glm.get_tstats(copes, varcopes))

        self.null_copes = np.reshape(null_copes, (self.n_perm, *self.glm.target_dims))
        self.null_tstats = np.reshape(null_tstats, (self.n_perm, *self.glm.target_dims))

    def _get_permute_feature_indx(self):
        """
        Get the indices of the features to permute.
        """
        return np.where(self.glm.c[self.contrast_indx] != 0.0)[0]

    def _validate_perm_type(self, perm_type):
        if perm_type is not None:
            if perm_type not in ["sign_flip", "row_shuffle"]:
                raise ValueError(
                    f"perm_type must be 'sign_flip' or 'row_shuffle', got {perm_type}"
                )
            return perm_type

        permute_indx = self._get_permute_feature_indx()
        feature_types = np.array(self.glm.feature_types)[permute_indx]
        feature_type = np.unique(feature_types)
        contrast_type = self.glm.contrast_types[self.contrast_indx]

        if len(feature_type) > 1:
            raise ValueError(
                f"Cannot determine perm_type when feature types are mixed. Got {feature_type}"
            )

        if feature_type == "constant":
            return "sign_flip"

        if feature_type == "categorical":
            if contrast_type == "differential":
                return "row_shuffle"
            return "sign_flip"

        return "row_shuffle"

    @property
    def copes(self):
        """
        COPEs of the contrast of interest.
        """
        return self.glm.copes[self.contrast_indx]

    @property
    def tstats(self):
        """
        T-stats of the contrast of interest.
        """
        return self.glm.tstats[self.contrast_indx]

    def summary(self):
        """
        Print summary of the permutation test.
        """
        sum = self.glm.summary()
        sum["n_perm"] = self.n_perm
        sum["perm_type"] = self.perm_type
        sum["contrast_indx"] = self.contrast_indx
        sum["n_jobs"] = self.n_jobs
        return sum


class MaxStatPermutation(Permutation):
    """
    Max statistic permutation test.

    Parameters
    ----------
    design : osl_dynamics.glm.base.Design
        Design object.
    contrast_indx : int
        Index of the contrast of interest.
    n_perm : int
        Number of permutations.
    perm_type : str, optional
        Type of permutation. Options are 'sign_flip' and 'row_shuffle'.
        If None, it will be determined based on the feature types and contrast type.
    n_jobs : int, optional
        Number of jobs to run in parallel. Default is 1.
    """

    def fit(self, y):
        """
        Fit the GLM with unpermuted data and run permutations.

        Parameters
        ----------
        y : np.ndarray
            Target variable. Shape is (n_samples, *target_dims).
        """
        super().fit(y)
        pool_axis = tuple(range(1, len(self.glm.target_dims) + 1))
        self.null_max_copes = np.nanmax(np.abs(self.null_copes), axis=pool_axis)
        self.null_max_tstats = np.nanmax(np.abs(self.null_tstats), axis=pool_axis)

    def get_pvalues(self, metric="copes"):
        """
        Get p-values.

        Parameters
        ----------
        metric : str, optional
            Metric to compute p-values. Options are 'copes' and 'tstats'.
            Default is 'copes'.

        Returns
        -------
        pvalues : np.ndarray
            P-values. Shape is (*target_dims,).
        """
        if metric == "copes":
            obs_stat = np.abs(self.glm.copes[self.contrast_indx])
            percentiles = stats.percentileofscore(self.null_max_copes, obs_stat)
        elif metric == "tstats":
            obs_stat = np.abs(self.glm.tstats[self.contrast_indx])
            percentiles = stats.percentileofscore(self.null_max_tstats, obs_stat)
        else:
            raise ValueError(f"metric must be 'copes' or 'tstats', got {metric}")

        return 1 - percentiles / 100
