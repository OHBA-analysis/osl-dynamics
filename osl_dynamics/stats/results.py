"""Result objects returned by the statistical tests."""

import logging
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats

from osl_dynamics.stats.correction import correct
from osl_dynamics.stats.design import Design

_logger = logging.getLogger("osl-dynamics")


def parametric_pvalues(
    tstats: np.ndarray, dof: np.ndarray, tail: int = 0
) -> np.ndarray:
    """P-values from the Student t distribution.

    Parameters
    ----------
    tstats : np.ndarray
        t-statistics.
    dof : np.ndarray
        Degrees of freedom, broadcastable to :code:`tstats`.
    tail : int, optional
        :code:`0` for two-tailed, :code:`1` for :code:`t > 0`, :code:`-1`
        for :code:`t < 0`.

    Returns
    -------
    pvalues : np.ndarray
        P-values. Same shape as :code:`tstats`.
    """
    if tail == 0:
        return 2 * scipy_stats.t.sf(np.abs(tstats), dof)
    if tail == 1:
        return scipy_stats.t.sf(tstats, dof)
    if tail == -1:
        return scipy_stats.t.cdf(tstats, dof)
    raise ValueError(f"tail must be -1, 0 or 1, got {tail}.")


def _tstats(copes: np.ndarray, varcopes: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return copes / np.sqrt(varcopes)


@dataclass
class GLMResult:
    """A fitted GLM.

    Parameters
    ----------
    design : Design
        The design that was fitted.
    betas : np.ndarray
        Regression coefficients. Shape is (n_features, \\*target_dims).
    copes : np.ndarray
        Contrast parameter estimates. Shape is (n_contrasts, \\*target_dims).
    varcopes : np.ndarray
        Variance of the copes. Shape is (n_contrasts, \\*target_dims).
    dof : np.ndarray
        Degrees of freedom of each target (they differ between targets only
        when there are missing values). Shape is (\\*target_dims).
    """

    design: Design
    betas: np.ndarray
    copes: np.ndarray
    varcopes: np.ndarray
    dof: np.ndarray

    @property
    def tstats(self) -> np.ndarray:
        """t-statistics. Shape is (n_contrasts, \\*target_dims)."""
        return _tstats(self.copes, self.varcopes)

    @property
    def target_dims(self) -> Tuple[int, ...]:
        return tuple(self.copes.shape[1:])

    @property
    def contrast_names(self) -> List[str]:
        return self.design.contrast_names

    def pvalues(self, tail: int = 0) -> np.ndarray:
        """Uncorrected parametric p-values for every contrast.

        Parameters
        ----------
        tail : int, optional
            :code:`0` for two-tailed, :code:`1` for :code:`t > 0`,
            :code:`-1` for :code:`t < 0`.

        Returns
        -------
        pvalues : np.ndarray
            Shape is (n_contrasts, \\*target_dims).
        """
        return parametric_pvalues(self.tstats, self.dof[None], tail)

    def test(
        self,
        contrast_indx: int = 0,
        correction: Optional[str] = None,
        tail: int = 0,
    ) -> "TestResult":
        """Parametric test of a contrast.

        Parameters
        ----------
        contrast_indx : int, optional
            Which contrast to test.
        correction : str, optional
            :code:`'bonferroni'`, :code:`'holm'`, :code:`'fdr'` or
            :code:`None` for uncorrected p-values.
        tail : int, optional
            :code:`0` for two-tailed, :code:`1` for :code:`t > 0`,
            :code:`-1` for :code:`t < 0`.

        Returns
        -------
        result : TestResult
            Test result with parametric p-values.
        """
        uncorrected = self.pvalues(tail)[contrast_indx]
        return TestResult(
            contrast=self.contrast_names[contrast_indx],
            copes=self.copes[contrast_indx],
            tstats=self.tstats[contrast_indx],
            pvalues=correct(uncorrected, correction),
            pvalues_uncorrected=uncorrected,
            correction=correction,
            tail=tail,
            metric="tstats",
            parametric=True,
            glm=self,
        )

    def summary(self) -> Dict:
        """Summary of the fit."""
        summary = self.design.summary()
        summary["target_dims"] = self.target_dims
        summary["dof"] = (int(self.dof.min()), int(self.dof.max()))
        return summary


@dataclass
class TestResult:
    """Result of a statistical test of one contrast.

    Parameters
    ----------
    contrast : str
        Name of the contrast tested.
    copes : np.ndarray
        Contrast parameter estimate (e.g. the group difference or mean
        response). Shape is (\\*target_dims).
    tstats : np.ndarray
        t-statistics. Shape is (\\*target_dims).
    pvalues : np.ndarray
        P-values after the multiple comparisons correction. Shape is
        (\\*target_dims). For :code:`correction='cluster'` every point holds
        the p-value of the cluster it belongs to (1 if it is in no cluster);
        for :code:`'tfce'` every point has its own p-value.
    correction : str or None
        Multiple comparisons correction that was applied.
    tail : int
        :code:`0` for two-tailed, :code:`1` or :code:`-1` for one-tailed.
    glm : GLMResult
        The fitted GLM (all contrasts).
    metric : str
        Statistic the permutation test was run on, :code:`'tstats'` or
        :code:`'copes'`.
    parametric : bool
        Whether the p-values come from the t distribution rather than
        permutations.
    pvalues_uncorrected : np.ndarray, optional
        Uncorrected p-values, when they were calculated.
    n_perm : int, optional
        Number of permutations used.
    perm_type : str, optional
        :code:`'sign_flip'` or :code:`'row_shuffle'`.
    exact : bool
        Whether every possible permutation was enumerated.
    pooled_dims : tuple of int, optional
        Target dimensions the maximum statistic was taken over.
    null : np.ndarray, optional
        Null distribution of the (reduced) statistic. Shape is
        (n_perm, \\*nonpooled_dims). :code:`None` if it was too large to keep.
    clusters : list of np.ndarray, optional
        Boolean masks of the clusters found in the observed data, largest
        statistic first. Shape of each is (\\*target_dims).
    cluster_stats : np.ndarray, optional
        Statistic of each cluster.
    cluster_pvalues : np.ndarray, optional
        P-value of each cluster.
    cluster_forming_threshold : float, optional
        Threshold used to form clusters.
    tfce_map : np.ndarray, optional
        Observed TFCE-enhanced statistic map for :code:`correction='tfce'`.
        Shape is (\\*target_dims).
    """

    contrast: str
    copes: np.ndarray
    tstats: np.ndarray
    pvalues: np.ndarray
    correction: Optional[str]
    tail: int
    glm: GLMResult
    metric: str = "tstats"
    parametric: bool = False
    pvalues_uncorrected: Optional[np.ndarray] = None
    n_perm: Optional[int] = None
    perm_type: Optional[str] = None
    exact: bool = False
    pooled_dims: Optional[Tuple[int, ...]] = None
    null: Optional[np.ndarray] = field(default=None, repr=False)
    clusters: Optional[List[np.ndarray]] = field(default=None, repr=False)
    cluster_stats: Optional[np.ndarray] = None
    cluster_pvalues: Optional[np.ndarray] = None
    cluster_forming_threshold: Optional[float] = None
    tfce_map: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def target_dims(self) -> Tuple[int, ...]:
        return tuple(self.pvalues.shape)

    @property
    def dof(self) -> np.ndarray:
        """Degrees of freedom of each target. Shape is (\\*target_dims)."""
        return self.glm.dof

    def significant(self, alpha: float = 0.05) -> np.ndarray:
        """Which targets are significant.

        Parameters
        ----------
        alpha : float, optional
            Significance level.

        Returns
        -------
        significant : np.ndarray
            Boolean array, :code:`pvalues < alpha`. Shape is (\\*target_dims).
            For cluster-based corrections this marks the points that belong
            to a significant cluster, which does not establish that the
            effect is confined to, or present at, every one of those points.
        """
        return self.pvalues < alpha

    def summary(self) -> Dict:
        """Summary of the test."""
        summary = {
            "contrast": self.contrast,
            "target_dims": self.target_dims,
            "correction": self.correction,
            "tail": self.tail,
            "parametric": self.parametric,
            "n_significant (p<0.05)": int(np.sum(self.pvalues < 0.05)),
            "min_pvalue": (
                float(np.nanmin(self.pvalues))
                if np.any(np.isfinite(self.pvalues))
                else np.nan
            ),
        }
        if not self.parametric:
            summary.update(
                {
                    "metric": self.metric,
                    "n_perm": self.n_perm,
                    "perm_type": self.perm_type,
                    "exact": self.exact,
                    "pooled_dims": self.pooled_dims,
                }
            )
        if self.clusters is not None:
            summary["n_clusters"] = len(self.clusters)
            summary["cluster_forming_threshold"] = self.cluster_forming_threshold
        return summary

    def __repr__(self) -> str:
        items = ", ".join(f"{k}={v!r}" for k, v in self.summary().items())
        return f"TestResult({items})"

    def save(self, filename: str) -> None:
        """Save to a pickle file.

        Parameters
        ----------
        filename : str
            Output file.
        """
        _logger.info(f"Saving {filename}")
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, filename: str) -> "TestResult":
        """Load from a pickle file.

        Parameters
        ----------
        filename : str
            File saved with :meth:`save`.

        Returns
        -------
        result : TestResult
        """
        with open(filename, "rb") as file:
            result = pickle.load(file)
        if not isinstance(result, cls):
            raise TypeError(f"{filename} does not contain a {cls.__name__}.")
        return result

    def plot_null(
        self,
        n_bins: int = 50,
        ax=None,
        filename: Optional[str] = None,
    ):
        """Plot the null distribution with the observed statistic.

        Only available for permutation tests that kept the null distribution.

        Parameters
        ----------
        n_bins : int, optional
            Number of histogram bins.
        ax : plt.Axes, optional
            Axis to plot on.
        filename : str, optional
            Output filename.

        Returns
        -------
        fig : plt.Figure
            Only returned if :code:`ax=None` and :code:`filename=None`.
        ax : plt.Axes
            Only returned if :code:`ax=None` and :code:`filename=None`.
        """
        if self.null is None:
            raise ValueError("This result has no null distribution to plot.")

        import matplotlib.pyplot as plt

        create_fig = ax is None
        if create_fig:
            fig, ax = plt.subplots(figsize=(6, 4))

        if self.correction == "cluster":
            stats = np.abs(self.cluster_stats) if self.cluster_stats is not None else []
            observed_max = np.max(stats) if len(stats) else 0.0
            label = "cluster statistic"
        elif self.correction == "tfce":
            observed_max = np.nanmax(np.abs(self.tfce_map))
            label = "TFCE"
        else:
            observed = self.tstats if self.metric == "tstats" else self.copes
            if self.tail == 0:
                observed = np.abs(observed)
            elif self.tail == -1:
                observed = -observed
            observed_max = np.nanmax(observed)
            label = "|{}|" if self.tail == 0 else "{}"
            label = label.format("t" if self.metric == "tstats" else "cope")
        null = self.null.reshape(-1)
        ax.hist(null[np.isfinite(null)], bins=n_bins, color="tab:grey")
        ax.axvline(observed_max, color="tab:red", label="observed max")
        ax.axvline(
            np.nanpercentile(null, 95), color="black", ls="--", label="95th percentile"
        )
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.set_title(f"Null distribution: {self.contrast}")
        ax.legend()

        if filename is not None:
            _logger.info(f"Saving {filename}")
            fig = ax.get_figure()
            fig.savefig(filename)
            plt.close(fig)
        elif create_fig:
            return fig, ax
