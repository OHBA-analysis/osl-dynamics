"""Source reconstruction."""

import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import nibabel as nib
from scipy.spatial import KDTree

import mne
from mne.beamformer._compute_beamformer import (
    Beamformer,
    _reduce_leadfield_rank,
    _sym_inv_sm,
)
from mne.beamformer._lcmv import _apply_lcmv
from mne.minimum_norm.inverse import (
    _check_depth,
    _check_reference,
    _get_vertno,
    _prepare_forward,
)
from mne.utils import logger as mne_logger

from osl_dynamics.utils.filenames import OSLFilenames
from osl_dynamics.utils.misc import system_call

from . import rhino


def lcmv_beamformer(
    fns: OSLFilenames,
    data: Optional[Union[str, mne.io.Raw, mne.Epochs]] = None,
    chantypes: Optional[Union[str, List[str]]] = None,
    data_cov: Optional[mne.Covariance] = None,
    noise_cov: Optional[mne.Covariance] = None,
    pick_ori: Optional[str] = "max-power-pre-weight-norm",
    rank: Union[str, Dict] = "info",
    noise_rank: Union[str, Dict] = "info",
    reduce_rank: bool = True,
    frequency_range: Optional[List[float]] = None,
    **kwargs,
) -> None:
    """Compute LCMV spatial filter.

    Parameters
    ----------
    fns : OSLFilenames
        Container for OSL filenames.
    data : str | instance of mne.Raw | mne.Epochs, optional
        The measurement data to specify the channels to include. Bad channels
        in info['bads'] are not used. Will also be used to calculate data_cov.
        If None, fns.preproc_file is used.
    chantypes : list or str
        List of channel types to use to calculate the noise covariance.
        E.g. ['eeg'], ['mag', 'grad'], ['eeg', 'mag', 'grad'].
    data_cov : instance of mne.Covariance | None
        The noise covariance matrix used to whiten.
        If None will be computed from data.
    noise_cov : instance of mne.Covariance | None
        The noise covariance matrix used to whiten.
        If None will be computed from data as a diagonal matrix with variances
        set to the average of all sensors of that type.
    pick_ori : None | 'normal' | 'max-power' | max-power-pre-weight-norm
        The source orientation to compute the beamformer in.
    rank : dict
        Calculate the rank only for a subset of channel types, and
        explicitly specify the rank for the remaining channel types.
        This can be extremely useful if you already know the rank of (part
        of) your data, for instance in case you have calculated it earlier.
        This parameter must be a dictionary whose keys correspond to channel
        types in the data (e.g. 'meg', 'mag', 'grad', 'eeg'), and whose
        values are integers representing the respective ranks. For example,
        {'mag': 90, 'eeg': 45} will assume a rank of 90 and 45 for
        magnetometer data and EEG data, respectively.
        The ranks for all channel types present in the data, but not
        specified in the dictionary will be estimated empirically. That is,
        if you passed a dataset containing magnetometer, gradiometer, and
        EEG data together with the dictionary from the previous example,
        only the gradiometer rank would be determined, while the specified
        magnetometer and EEG ranks would be taken for granted.
    noise_rank : dict | None | 'full' | 'info'
        This controls the rank computation that can be read from the measurement
        info or estimated from the data. When a noise covariance is used for
        whitening, this should reflect the rank of that covariance, otherwise
        amplification of noise components can occur in whitening (e.g., often
        during source localization).
    reduce_rank : bool
        Whether to reduce the rank by one during computation of the filter.
    frequency_range : list
        Lower and upper range (in Hz) for frequency range to bandpass filter.
        If None, no filtering is performed.
    **kwargs : keyword arguments
        Keyword arguments that will be passed to mne.beamformer.make_lcmv.

    Returns
    -------
    filters : instance of mne.beamformer.Beamformer
        Dictionary containing filter weights from LCMV beamformer.
        See: https://mne.tools/stable/generated/mne.beamformer.make_lcmv.html
    """
    print("")
    print("Making LCMV beamformer")
    print("----------------------")

    if data is None:
        data = fns.preproc_file

    if chantypes is None:
        raise ValueError("chantypes must be passed.")

    if isinstance(chantypes, str):
        chantypes = [chantypes]

    # Load data
    if isinstance(data, str):
        if "epo.fif" in data:
            data = mne.read_epochs(data)
        else:
            data = mne.io.read_raw_fif(data)

    # Bandpass filter
    if frequency_range is not None:
        data.filter(
            l_freq=frequency_range[0],
            h_freq=frequency_range[1],
            method="iir",
            iir_params={"order": 5, "ftype": "butter"},
        )

    # Load forward solution
    fwd = mne.read_forward_solution(fns.fwd_model)

    if data_cov is None:
        # Note that if chantypes are meg, eeg; and meg includes mag, grad then
        # compute_covariance will project data separately for meg and eeg to
        # reduced rank subspace (i.e. mag and grad will be combined together
        # inside the meg subspace, eeg will haeve a separate subspace). I.e.
        # data will become (ntpts x (rank_meg + rank_eeg)) and cov will be
        # (rank_meg + rank_eeg) x (rank_meg + rank_eeg) and include correlations
        # between eeg and meg subspaces. The output data_cov is cov projected
        # back onto the indivdual sensor types mag, grad, eeg.
        #
        # Prior to computing anything, including the subspaces each of mag, grad,
        # eeg are scaled so that they are on comparable scales to aid mixing in
        # the subspace and improve numerical stability. Note that in the
        # output data_cov the scalings have been undone.
        if isinstance(data, mne.Epochs):
            data_cov = mne.compute_covariance(data, method="empirical", rank=rank)
        else:
            data_cov = mne.compute_raw_covariance(data, method="empirical", rank=rank)

    if noise_cov is None:
        # Calculate noise covariance matrix
        #
        # Later this will be inverted and used to whiten the data AND the lead
        # fields as part of the source recon. See:
        #
        #   https://www.sciencedirect.com/science/article/pii/S1053811914010325
        #
        # In MNE, the noise cov is normally obtained from empty room noise
        # recordings or from a baseline period. Here (if no noise cov is passed
        # in) we compute a diagonal noise cov with the variances set to the mean
        # variance of each sensor type (e.g. mag, grad, eeg).
        n_channels = data_cov.data.shape[0]
        noise_cov_diag = np.zeros(n_channels)
        for type in chantypes:
            # Indices of this channel type
            type_data = data.copy().pick(type, exclude="bads")
            inds = []
            for chan in type_data.info["ch_names"]:
                inds.append(data_cov.ch_names.index(chan))

            # Mean variance of channels of this type
            variance = np.mean(np.diag(data_cov.data)[inds])
            noise_cov_diag[inds] = variance
            print(f"Variance for chantype {type} is {variance}")

        bads = [b for b in data.info["bads"] if b in data_cov.ch_names]
        noise_cov = mne.Covariance(
            noise_cov_diag,
            data_cov.ch_names,
            bads,
            data.info["projs"],
            nfree=1e10,
        )

    # Make filters
    filters = _make_lcmv(
        data.info,
        fwd,
        data_cov,
        noise_cov=noise_cov,
        pick_ori=pick_ori,
        rank=rank,
        noise_rank=noise_rank,
        reduce_rank=reduce_rank,
        **kwargs,
    )

    print(f"Saving {fns.filters}")
    filters.save(fns.filters, overwrite=True)

    print("LCMV beamformer complete.")


def apply_lcmv_beamformer(
    fns: OSLFilenames,
    raw: Optional[Union[mne.io.Raw, mne.Epochs]] = None,
    reject_by_annotation: Optional[Union[str, List[str]]] = "omit",
    spatial_resolution: Optional[int] = None,
    reference_brain: str = "mni",
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply an LCMV beamformer.

    Parameters
    ----------
    fns : OSLFilenames
        Container for OSL filenames.
    raw : instance of mne.io.Raw or mne.Epochs, optional
        The data to apply the LCMV filter to.
        If None, fns.preproc_file is used.
    reject_by_annotation : str | list of str | None
        If string, the annotation description to use to reject epochs.
        If list of str, the annotation descriptions to use to reject epochs.
        If None, do not reject epochs.
    spatial_resolution : int, optional
        Resolution to use for the reference brain in mm (must be an integer,
        or will be cast to nearest int). If None, then the gridstep used to
        create the forward model is used.
    reference_brain : str, optional
        Either 'head' or 'mni'.

    Returns
    -------
    voxel_data : np.ndarray
        Voxel data in the space specified by the 'reference_brain' argument.
        Shape is (voxels, time) for a Raw object or (voxels, time, epochs) for
        an Epochs object.
    coords : np.ndarray
        Coordinates for each voxel in the reference brain space.
        Shape is (voxels, 3).
    """
    print()
    print("Applying LCMV beamformer")
    print("------------------------")

    if raw is None:
        raw = mne.io.read_raw_fif(fns.preproc_file, preload=True)

    # Load filters
    filters = mne.beamformer.read_beamformer(fns.filters)

    # Pick chantypes that were used to make the beamformer in the data
    raw = raw.copy().pick(filters["ch_names"])

    if isinstance(raw, mne.Epochs):
        # Apply filters to an Epochs object
        stc = mne.beamformer.apply_lcmv_epochs(raw, filters)
        voxel_data_head = np.transpose([s.data for s in stc], axes=[1, 2, 0])
    else:
        # Apply filters to a Raw object
        _check_reference(raw)
        data, times = raw.get_data(
            return_times=True, reject_by_annotation=reject_by_annotation
        )
        chan_inds = mne.utils._check_channels_spatial_filter(raw.ch_names, filters)
        data = data[chan_inds]
        stc = _apply_lcmv(data=data, filters=filters, info=raw.info, tmin=times[0])
        voxel_data_head = next(stc).data

    # Get coordinates in head space
    fwd = mne.read_forward_solution(fns.fwd_model)
    vs = fwd["src"][0]
    voxel_coords_head = vs["rr"][vs["vertno"]] * 1000  # in mm

    if reference_brain == "head":
        return voxel_data_head, voxel_coords_head

    # ------------------------------------------
    # Convert coordinates from head space to MNI
    # ------------------------------------------

    # Convert voxel_coords_head to unscaled MRI
    # head_mri_t_file xform is to unscaled MRI
    head_mri_t = mne.transforms.read_trans(fns.coreg.head_mri_t_file)
    voxel_coords_mri = rhino._xform_points(
        head_mri_t["trans"], voxel_coords_head.T
    ).T

    # Convert voxel_coords_mri to MNI
    # mni_mri_t_file xform is to unscaled MRI
    mni_mri_t = mne.transforms.read_trans(fns.surfaces.mni_mri_t_file)
    voxel_coords_mni = rhino._xform_points(
        np.linalg.inv(mni_mri_t["trans"]), voxel_coords_mri.T
    ).T

    if spatial_resolution is None:
        # Estimate gridstep from forward model
        rr = fwd["src"][0]["rr"]
        spatial_resolution = _get_gridstep(rr)

    spatial_resolution = int(spatial_resolution)
    print(f"spatial_resolution = {spatial_resolution} mm")

    reference_brain = f"{fns.surfaces.fsl_dir}/data/standard/MNI152_T1_1mm_brain.nii.gz"

    # Create standard brain of the required resolution
    reference_brain_resampled = (
        f"{fns.src_dir}/MNI152_T1_{spatial_resolution}mm_brain.nii.gz"
    )
    print(f"mask_file: {reference_brain_resampled}")

    # Get coordinates from reference brain at resolution spatial_resolution
    system_call(
        f"flirt -in {reference_brain} -ref {reference_brain} "
        f"-out {reference_brain_resampled} -applyisoxfm {spatial_resolution}",
        verbose=False
    )
    voxel_coords_mni_resampled, _ = _niimask2mmpointcloud(reference_brain_resampled)

    # For each voxel_coords_mni find nearest coordinate in voxel_coords_head
    print("Finding nearest neighbour in resampled MNI space")
    voxel_data_mni_resampled = np.zeros(
        np.insert(voxel_data_head.shape[1:], 0, voxel_coords_mni_resampled.shape[1])
    )
    for cc in range(voxel_coords_mni_resampled.shape[1]):
        index, dist = _closest_node(voxel_coords_mni_resampled[:, cc], voxel_coords_mni)
        if dist < spatial_resolution:
            voxel_data_mni_resampled[cc] = voxel_data_head[index]

    print("Applying LCMV beamformer complete.")

    return voxel_data_mni_resampled, voxel_coords_mni_resampled


def _make_lcmv(
    info,
    fwd,
    data_cov,
    reg=0.05,
    noise_cov=None,
    label=None,
    pick_ori=None,
    rank="info",
    noise_rank="info",
    weight_norm="unit-noise-gain-invariant",
    reduce_rank=False,
    depth=None,
    inversion="matrix",
    verbose=None,
):
    """Compute LCMV spatial filter.

    Modified version of mne.beamformer.make_lcmv.
    
    Parameters
    ----------
    info : instance of mne.Info
        The measurement info to specify the channels to include.
    fwd : instance of mne.Forward
        The fwd solution.
    data_cov : instance of mne.Covariance
        The data covariance object.
    reg : float
        The regularization for the whitened data covariance.
    noise_cov : instance of mne.Covariance
        The noise covariance object.
    label : instance of mne.Label
        Restricts the LCMV solution to a given label.
    pick_ori : None | 'normal' | 'max-power' | max-power-pre-weight-norm
        The source orientation to compute the beamformer in.
    rank : dict
        Calculate the rank only for a subset of channel types, and
        explicitly specify the rank for the remaining channel types.
        This can be extremely useful if you already know the rank of (part
        of) your data, for instance in case you have calculated it earlier.
        This parameter must be a dictionary whose keys correspond to channel
        types in the data (e.g. 'meg', 'mag', 'grad', 'eeg'), and whose
        values are integers representing the respective ranks. For example,
        {'mag': 90, 'eeg': 45} will assume a rank of 90 and 45 for
        magnetometer data and EEG data, respectively.
        The ranks for all channel types present in the data, but not
        specified in the dictionary will be estimated empirically. That is,
        if you passed a dataset containing magnetometer, gradiometer, and
        EEG data together with the dictionary from the previous example,
        only the gradiometer rank would be determined, while the specified
        magnetometer and EEG ranks would be taken for granted.
    noise_rank : dict
        This controls the rank computation that can be read from the measurement
        info or estimated from the data. When a noise covariance is used for
        whitening, this should reflect the rank of that covariance, otherwise
        amplification of noise components can occur in whitening (e.g., often
        during source localization).
    weight_norm : None | 'unit-noise-gain' | 'nai'
        The weight normalization scheme to use.
    reduce_rank : bool
        Whether to reduce the rank by one during computation of the filter.
    depth : None | float | dict
        How to weight (or normalize) the forward using a depth prior (see Notes).
    inversion : 'matrix' | 'single'
        The inversion scheme to compute the weights.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
        
    Returns
    -------
    filters : instance of mne.beamformer.Beamformer
        Dictionary containing filter weights from LCMV beamformer. See MNE docs.
        
    """
    # Check number of sensor types present in the data and ensure a noise cov
    info = mne._fiff.meas_info._simplify_info(info)
    noise_cov, _, allow_mismatch = mne.utils._check_one_ch_type(
        "lcmv", info, fwd, data_cov, noise_cov
    )

    # NOTE: we need this extra picking step (can't just rely on minimum
    # norm's because there can be a mismatch. Should probably add an extra
    # arg to _prepare_beamformer_input at some point (later)
    picks = mne.utils._check_info_inv(info, fwd, data_cov, noise_cov)
    info = mne._fiff.pick.pick_info(info, picks)

    data_rank = mne.rank.compute_rank(data_cov, rank=rank, info=info)
    noise_rank = mne.rank.compute_rank(noise_cov, rank=noise_rank, info=info)

    mne_logger.info(f"Making LCMV beamformer with data cov rank {data_rank}")
    mne_logger.info(f"Making LCMV beamformer with noise cov rank {noise_rank}")

    depth = _check_depth(depth, "depth_sparse")
    if inversion == "single":
        depth["combine_xyz"] = False

    is_free_ori, info, proj, vertno, G, whitener, nn, orient_std = _prepare_beamformer_input(
        info,
        fwd,
        label,
        pick_ori,
        noise_cov=noise_cov,
        rank=noise_rank,
        pca=False,
        **depth,
    )
    del noise_rank

    ch_names = list(info["ch_names"])

    data_cov = mne._fiff.pick.pick_channels_cov(data_cov, include=ch_names)
    Cm = data_cov._get_square()
    if "estimator" in data_cov:
        del data_cov["estimator"]
    rank_int = sum(data_rank.values())
    del data_rank

    # Compute spatial filter
    n_orient = 3 if is_free_ori else 1
    W, max_power_ori = _compute_beamformer(
        G,
        Cm,
        reg,
        n_orient,
        weight_norm,
        pick_ori,
        reduce_rank,
        rank_int,
        inversion=inversion,
        nn=nn,
        orient_std=orient_std,
        whitener=whitener,
    )

    # Get src type to store with filters for _make_stc
    src_type = mne.source_estimate._get_src_type(fwd["src"], vertno)

    # Get subject to store with filters
    subject_from = mne.forward._subject_from_forward(fwd)

    # Is the computed beamformer a scalar or vector beamformer?
    is_free_ori = is_free_ori if pick_ori in [None, "vector"] else False
    is_ssp = bool(info["projs"])

    filters = Beamformer(
        kind="LCMV",
        weights=W,
        data_cov=data_cov,
        noise_cov=noise_cov,
        whitener=whitener,
        weight_norm=weight_norm,
        pick_ori=pick_ori,
        ch_names=ch_names,
        proj=proj,
        is_ssp=is_ssp,
        vertices=vertno,
        is_free_ori=is_free_ori,
        n_sources=fwd["nsource"],
        src_type=src_type,
        source_nn=fwd["source_nn"].copy(),
        subject=subject_from,
        rank=rank_int,
        max_power_ori=max_power_ori,
        inversion=inversion,
    )

    return filters


def _compute_beamformer(
    G,
    Cm,
    reg,
    n_orient,
    weight_norm,
    pick_ori,
    reduce_rank,
    rank,
    inversion,
    nn,
    orient_std,
    whitener,
):
    """Compute a spatial beamformer filter (LCMV or DICS).

    Modified version of mne.beamformer._compute_beamformer.

    Parameters
    ----------
    G : (n_dipoles, n_channels) numpy.ndarray
        The leadfield.
    Cm : (n_channels, n_channels) numpy.ndarray
        The data covariance matrix.
    reg : float
        Regularization parameter.
    n_orient : int
        Number of dipole orientations defined at each source point
    weight_norm : None | 'unit-noise-gain' | 'nai'
        The weight normalization scheme to use.
    pick_ori : None | 'normal' | 'max-power' | max-power-pre-weight-norm
        The source orientation to compute the beamformer in.
    reduce_rank : bool
        Whether to reduce the rank by one during computation of the filter.
    rank : dict | None | 'full' | 'info'
        See compute_rank.
    inversion : 'matrix' | 'single'
        The inversion scheme to compute the weights.
    nn : (n_dipoles, 3) numpy.ndarray
        The source normals.
    orient_std : (n_dipoles,) numpy.ndarray
        The std of the orientation prior used in weighting the lead fields.
    whitener : (n_channels, n_channels) numpy.ndarray
        The whitener.

    For more detailed information on the parameters, see the docstrings of
    `make_lcmv` and `make_dics`.

    Returns
    -------
    W : (n_dipoles, n_channels) numpy.ndarray
        The beamformer filter weights.
    """
    # Lines changes are marked with MWW

    mne.utils._check_option(
        "weight_norm",
        weight_norm,
        ["unit-noise-gain-invariant", "unit-noise-gain", "nai", None],
    )

    # Whiten the data covariance
    Cm = whitener @ Cm @ whitener.T.conj()

    # Restore to properly Hermitian as large whitening coefs can have bad
    # rounding error
    Cm[:] = (Cm + Cm.T.conj()) / 2.0

    assert Cm.shape == (G.shape[0],) * 2

    s, _ = np.linalg.eigh(Cm)
    if not (s >= -s.max() * 1e-7).all():
        # This shouldn't ever happen, but just in case
        warn(
            "data covariance does not appear to be positive semidefinite, "
            "results will likely be incorrect"
        )

    # Tikhonov regularization using reg parameter to control for trade-off
    # between spatial resolution and noise sensitivity
    # eq. 25 in Gross and Ioannides, 1999 Phys. Med. Biol. 44 2081
    Cm_inv, loading_factor, rank = mne.utils._reg_pinv(Cm, reg, rank)

    assert orient_std.shape == (G.shape[1],)

    n_sources = G.shape[1] // n_orient

    assert nn.shape == (n_sources, 3)

    mne_logger.info(
        f"Computing beamformer filters for {n_sources} source{mne.utils._pl(n_sources)}"
    )
    n_channels = G.shape[0]

    assert n_orient in (3, 1)

    Gk = np.reshape(G.T, (n_sources, n_orient, n_channels)).transpose(0, 2, 1)

    assert Gk.shape == (n_sources, n_channels, n_orient)

    sk = np.reshape(orient_std, (n_sources, n_orient))

    del G, orient_std

    mne.utils._check_option("reduce_rank", reduce_rank, (True, False))

    # inversion of the denominator
    mne.utils._check_option("inversion", inversion, ("matrix", "single"))

    if (
        inversion == "single" \
        and n_orient > 1 \
        and pick_ori == "vector" \
        and weight_norm == "unit-noise-gain-invariant"
    ):
        raise ValueError(
            'Cannot use pick_ori="vector" with inversion="single" and '
            'weight_norm="unit-noise-gain-invariant"'
        )

    if reduce_rank and inversion == "single":
        raise ValueError(
            'reduce_rank cannot be used with inversion="single"; '
            'consider using inversion="matrix" if you have a rank-deficient '
            'forward model (i.e., from a sphere model with MEG channels), '
            'otherwise consider using reduce_rank=False'
        )
    if n_orient > 1:
        _, Gk_s, _ = np.linalg.svd(Gk, full_matrices=False)
        assert Gk_s.shape == (n_sources, n_orient)
        if not reduce_rank and (Gk_s[:, 0] > 1e6 * Gk_s[:, 2]).any():
            raise ValueError(
                "Singular matrix detected when estimating spatial filters. "
                "Consider reducing the rank of the forward operator by using "
                "reduce_rank=True."
            )
        del Gk_s

    # --------------------------------
    # 1. Reduce rank of the lead field
    # --------------------------------

    if reduce_rank:
        Gk = _reduce_leadfield_rank(Gk)

    def _compute_bf_terms(Gk, Cm_inv):
        bf_numer = np.matmul(Gk.swapaxes(-2, -1).conj(), Cm_inv)
        bf_denom = np.matmul(bf_numer, Gk)
        return bf_numer, bf_denom

    # ----------------------------------------------------------
    # 2. Reorient lead field in direction of max power or normal
    # ----------------------------------------------------------

    if pick_ori == "max-power" or pick_ori == "max-power-pre-weight-norm":
        assert n_orient == 3
        _, bf_denom = _compute_bf_terms(Gk, Cm_inv)

        if pick_ori == "max-power":
            if weight_norm is None:
                ori_numer = np.eye(n_orient)[np.newaxis]
                ori_denom = bf_denom
            else:
                # Compute power, cf Sekihara & Nagarajan 2008, eq. 4.47
                ori_numer = bf_denom

                # Cm_inv should be Hermitian so no need for .T.conj()
                ori_denom = np.matmul(
                    np.matmul(Gk.swapaxes(-2, -1).conj(), Cm_inv @ Cm_inv), Gk
                )

            ori_denom_inv = _sym_inv_sm(ori_denom, reduce_rank, inversion, sk)
            ori_pick = np.matmul(ori_denom_inv, ori_numer)

        else:
            # MWW
            #
            # Compute orientation based on pre-normalised power
            #
            # See eq 5 in Brookes et al, Optimising experimental design for
            # MEG beamformer imaging, Neuroimage 2008. This optimises the
            # orientation by maximising the power BEFORE any weight
            # normalisation is performed
            ori_pick = _sym_inv_sm(bf_denom, reduce_rank, inversion, sk)

        assert ori_pick.shape == (n_sources, n_orient, n_orient)

        # Pick eigenvector that corresponds to maximum eigenvalue
        eig_vals, eig_vecs = np.linalg.eig(ori_pick.real)  # not Hermitian!

        # Sort eigenvectors by eigenvalues for picking
        order = np.argsort(np.abs(eig_vals), axis=-1)
        #eig_vals = np.take_along_axis(eig_vals, order, axis=-1)
        max_power_ori = eig_vecs[np.arange(len(eig_vecs)), :, order[:, -1]]

        assert max_power_ori.shape == (n_sources, n_orient)

        # Set the (otherwise arbitrary) sign to match the normal
        signs = np.sign(np.sum(max_power_ori * nn, axis=1, keepdims=True))
        signs[signs == 0] = 1.0
        max_power_ori *= signs

        # Compute the lead field for the optimal orientation,
        # and adjust numer/denom
        Gk = np.matmul(Gk, max_power_ori[..., np.newaxis])
        n_orient = 1

    else:
        max_power_ori = None
        if pick_ori == "normal":
            Gk = Gk[..., 2:3]
            n_orient = 1

    # ----------------------------------------------------------------------
    # 3. Compute numerator and denominator of beamformer formula (unit-gain)
    # ----------------------------------------------------------------------

    bf_numer, bf_denom = _compute_bf_terms(Gk, Cm_inv)

    assert bf_denom.shape == (n_sources,) + (n_orient,) * 2
    assert bf_numer.shape == (n_sources, n_orient, n_channels)

    del Gk  # lead field has been adjusted and should not be used anymore

    # -------------------------
    # 4. Invert the denominator
    # -------------------------

    # Here W is W_ug, i.e.: G.T @ Cm_inv / (G.T @ Cm_inv @ G)
    bf_denom_inv = _sym_inv_sm(bf_denom, reduce_rank, inversion, sk)

    assert bf_denom_inv.shape == (n_sources, n_orient, n_orient)

    W = np.matmul(bf_denom_inv, bf_numer)

    assert W.shape == (n_sources, n_orient, n_channels)

    del bf_denom_inv, sk

    # ----------------------------------------------------------------
    # 5. Re-scale filter weights according to the selected weight_norm
    # ----------------------------------------------------------------

    # Weight normalization is done by computing, for each source:
    #
    #     W_ung = W_ug / sqrt(W_ug @ W_ug.T)
    #
    # with W_ung referring to the unit-noise-gain (weight normalized) filter
    # and W_ug referring to the above-calculated unit-gain filter stored in W.

    if weight_norm is not None:
        # Three different ways to calculate the normalization factors here.
        # Only matters when in vector mode, as otherwise n_orient == 1 and
        # they are all equivalent.
        #
        # In MNE < 0.21, we just used the Frobenius matrix norm:
        #
        #    noise_norm = np.linalg.norm(W, axis=(1, 2), keepdims=True)
        #    assert noise_norm.shape == (n_sources, 1, 1)
        #    W /= noise_norm
        #
        # Sekihara 2008 says to use sqrt(diag(W_ug @ W_ug.T)),
        # which is not rotation invariant:
        if weight_norm in ("unit-noise-gain", "nai"):
            noise_norm = np.matmul(W, W.swapaxes(-2, -1).conj()).real
            noise_norm = np.reshape(
                noise_norm, (n_sources, -1, 1)
            )[:, :: n_orient + 1] # np.diag operation over last two axes
            np.sqrt(noise_norm, out=noise_norm)
            noise_norm[noise_norm == 0] = np.inf
            assert noise_norm.shape == (n_sources, n_orient, 1)
            W /= noise_norm
        else:
            assert weight_norm == "unit-noise-gain-invariant"
            # Here we use sqrtm. The shortcut:
            #
            #    use = W
            #
            # ... does not match the direct route (it is rotated!),
            # so we'll use the direct one to match FieldTrip:
            use = bf_numer
            inner = np.matmul(use, use.swapaxes(-2, -1).conj())
            W = np.matmul(mne.utils._sym_mat_pow(inner, -0.5), use)
            noise_norm = 1.0

        if weight_norm == "nai":
            # Estimate noise level based on covariance matrix, taking the
            # first eigenvalue that falls outside the signal subspace
            # or the loading factor used during regularization, whichever
            # is largest.
            if rank > len(Cm):
                # Covariance matrix is full rank, no noise subspace!
                # Use the loading factor as noise ceiling.
                if loading_factor == 0:
                    raise RuntimeError(
                        "Cannot compute noise subspace with a full-rank "
                        "covariance matrix and no regularization. "
                        "Try manually specifying the rank of the covariance "
                        "matrix or using regularization."
                    )
                noise = loading_factor
            else:
                noise, _ = np.linalg.eigh(Cm)
                noise = noise[-rank]
                noise = max(noise, loading_factor)
            W /= np.sqrt(noise)

    W = W.reshape(n_sources * n_orient, n_channels)
    mne_logger.info("Filter computation complete")

    return W, max_power_ori


def _prepare_beamformer_input(
    info,
    forward,
    label=None,
    pick_ori=None,
    noise_cov=None,
    rank=None,
    pca=False,
    loose=None,
    combine_xyz="fro",
    exp=None,
    limit=None,
    allow_fixed_depth=True,
    limit_depth_chs=False,
):
    """Input preparation common for LCMV, DICS, and RAP-MUSIC.

    Modified version of mne.beamformer._prepare_beamformer_input.
    
    Parameters
    ----------
    info : instance of mne.Info
        Measurement info
    forward : instance of mne.Forward
        The forward solution.
    label : instance of mne.Label | None
        Restricts the forward solution to a given label.
    pick_ori : None | 'normal' | 'max-power' | 'vector' | 'max-power-pre-weight-norm'
        The source orientation to compute the beamformer in.
    noise_cov : instance of mne.Covariance | None
        The noise covariance.
    rank : dict | None | 'full' | 'info'
        See :py:func:`mne.compute_rank`.
    pca : bool
        If True, the rank of the forward is reduced to match the rank of the
        noise covariance matrix.
    loose : float | None
        Value that weights the source variances of the dipole components
        defining the tangent space of the cortical surfaces. If ``None``,
        no loose orientation constraint is applied.
    combine_xyz : str
        How to combine the dipoles in the same locations of the forward model
        when picking normals. See :py:func:`mne.forward._pick_ori`.
    exp : float | None
        Exponent for the depth weighting. If None, no depth weighting is performed.
    limit : float | None
        Limit on depth weighting factors. If None, no upper limit is applied.
    allow_fixed_depth : bool
        If True, fixed depth weighting is allowed.
    limit_depth_chs : bool
        If True, use only grad channels for depth weighting. 
        
    Returns
    -------
    is_free_ori : bool
        Whether the forward operator is free orientation.
    info : instance of mne.Info
        Measurement info restricted to selected channels.
    proj : array
        The SSP/PCA projector.
    vertno : array
        The indices of the vertices corresponding to the source space.
    G : array
        The forward operator restricted to selected channels.
    whitener : array  
        The whitener for the selected channels.
    nn : array
        The normals of the source space.
    orient_std : array
        The standard deviation of the orientation prior.
    """
    # Lines marked MWW are where code has been changed.

    # MWW
    # _check_option('pick_ori', pick_ori, ('normal', 'max-power', 'vector', None))
    mne.utils._check_option(
        "pick_ori",
        pick_ori,
        ("normal", "max-power", "vector", "max-power-pre-weight-norm", None),
    )

    # MWW
    # Restrict forward solution to selected vertices
    #if label is not None:
    #    _, src_sel = label_src_vertno_sel(label, forward["src"])
    #    forward = _restrict_forward_to_src_sel(forward, src_sel)

    if loose is None:
        loose = 0.0 if mne.forward.forward.is_fixed_orient(forward) else 1.0

    # MWW
    #if noise_cov is None:
    #    noise_cov = make_ad_hoc_cov(info, std=1.0)

    forward, info_picked, gain, _, orient_prior, _, trace_GRGT, noise_cov, whitener = _prepare_forward(
        forward,
        info,
        noise_cov,
        "auto",
        loose,
        rank=rank,
        pca=pca,
        use_cps=True,
        exp=exp,
        limit_depth_chs=limit_depth_chs,
        combine_xyz=combine_xyz,
        limit=limit,
        allow_fixed_depth=allow_fixed_depth,
    )
    is_free_ori = not mne.forward.forward.is_fixed_orient(forward)  # could have been changed
    nn = forward["source_nn"]
    if is_free_ori:  # take Z coordinate
        nn = nn[2::3]
    nn = nn.copy()
    vertno = _get_vertno(forward["src"])
    if forward["surf_ori"]:
        nn[...] = [0, 0, 1]  # align to local +Z coordinate
    if pick_ori is not None and not is_free_ori:
        raise ValueError(
            f"Normal or max-power orientation (got {pick_ori}) can only be "
            "picked when a forward operator with free orientation is used."
        )
    if pick_ori == "normal" and not forward["surf_ori"]:
        raise ValueError(
            "Normal orientation can only be picked when a forward operator "
            "oriented in surface coordinates is used."
        )
    mne.utils._check_src_normal(pick_ori, forward["src"])
    del forward, info

    # Undo the scaling that MNE prefers
    scale = np.sqrt((noise_cov["eig"] > 0).sum() / trace_GRGT)
    gain /= scale
    if orient_prior is not None:
        orient_std = np.sqrt(orient_prior)
    else:
        orient_std = np.ones(gain.shape[1])

    # Get the projector
    proj, _, _ = mne._fiff.proj.make_projector(
        info_picked["projs"], info_picked["ch_names"]
    )

    return is_free_ori, info_picked, proj, vertno, gain, whitener, nn, orient_std


def _niimask2mmpointcloud(
    nii_mask: str, volindex: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Takes in a nii.gz mask (which equals zero for background and neq zero
    for the mask) and returns the mask as a 3 x npoints point cloud in native
    space in mm's.

    Parameters
    ----------
    nii_mask : string
        A nii.gz mask file name or the [x,y,z] volume (with zero for background,
        and !=0 for the mask).
    volindex : int
        Volume index, used if nii_mask is a 4D file.

    Returns
    -------
    pc : numpy.ndarray
        3 x npoints point cloud as mm in native space (using sform).
    values : numpy.ndarray
        npoints values.
    """
    vol = nib.load(nii_mask).get_fdata()
    if len(vol.shape) == 4 and volindex is not None:
        vol = vol[:, :, :, volindex]
    if not len(vol.shape) == 3:
        Exception(
            "nii_mask must be a 3D volume, or nii_mask must be a 4D volume "
            "with volindex specifying a volume index"
        )
    pc_nativeindex = np.asarray(np.where(vol != 0))
    values = np.asarray(vol[vol != 0])
    pc = rhino._xform_points(rhino._get_sform(nii_mask)["trans"], pc_nativeindex)
    return pc, values


def _closest_node(
    node: np.ndarray, nodes: np.ndarray,
) -> Tuple[int, float]:
    """Find nearest node in nodes to the passed in node.

    Returns
    -------
    index : int
        Index to the nearest node in nodes.
    distance : float
        Distance.
    """
    if len(nodes) == 1:
        nodes = np.reshape(nodes, [-1, 1])
    kdtree = KDTree(nodes)
    distance, index = kdtree.query(node)
    return index, distance


def _get_gridstep(coords: np.ndarray) -> int:
    """Get gridstep (i.e. spatial resolution of dipole grid) in mm.

    Parameters
    ----------
    coords : numpy.ndarray
        Coordinates.

    Returns
    -------
    gridstep: int
        Spatial resolution of dipole grid in mm.
    """
    store = []
    for ii in range(coords.shape[0]):
        store.append(np.sqrt(np.sum(np.square(coords[ii, :] - coords[0, :]))))
    store = np.asarray(store)
    return int(np.round(np.min(store[np.where(store > 0)]) * 1000))
