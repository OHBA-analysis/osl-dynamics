"""Adaptive Multipole Model (AMM) for OPM interference rejection.

Consolidates spherical harmonics, spheroid fitting, prolate coordinate
transforms, internal/external harmonic bases, and the AMM denoising pipeline.

Translated from spm_opm_amm.m and supporting SPM functions.

References
----------
Tierney, T.M., Seedat, Z., St Pier, K. et al. (2024). Adaptive multipole
models of optically pumped magnetometer data. Human Brain Mapping, 45,
e26596. https://doi.org/10.1002/hbm.26596
"""

import mne
import numpy as np
from scipy.special import gammaln


def associated_legendre(x, l, m):
    """Compute associated Legendre polynomial P_l^m(x).

    Uses (-1)^m Condon-Shortley phase, matching spm_slm.m lines 64-76.

    Parameters
    ----------
    x : ndarray, shape (n,)
        cos(theta) values.
    l : int
        Degree.
    m : int
        Order (non-negative).

    Returns
    -------
    pl : ndarray, shape (n,)
    """
    b = (-1) ** m * 2**l
    pl = np.zeros_like(x, dtype=float)
    xsq = (1 - x**2) ** (m / 2)

    for k in range(m, l + 1):
        tmp = (l + k - 1) / 2 - np.arange(l)
        val = np.prod(tmp) if len(tmp) > 0 else 1.0
        vals2 = np.prod(l - np.arange(k)) if k > 0 else 1.0
        log_c = (
            gammaln(k + 1)
            - gammaln(k - m + 1)
            + np.log(np.abs(vals2) + 1e-300)
            - gammaln(k + 1)
            + np.log(np.abs(val) + 1e-300)
            - gammaln(l + 1)
        )
        sign_c = np.sign(vals2) * np.sign(val)
        c = sign_c * np.exp(log_c)
        pl = pl + b * xsq * c * x ** (k - m)

    return pl


def associated_legendre_deriv(theta, l, m):
    """Compute dP_l^m/dtheta.

    Translated from spm_slm.m lines 50-62.

    Parameters
    ----------
    theta : ndarray, shape (n,)
        Colatitude in radians.
    l : int
        Degree.
    m : int
        Order (non-negative).

    Returns
    -------
    dpl : ndarray, shape (n,)
    """
    b = (-1) ** m * 2**l
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    dpl = np.zeros_like(theta, dtype=float)

    for k in range(m, l + 1):
        tmp = (l + k - 1) / 2 - np.arange(l)
        val = np.prod(tmp) if len(tmp) > 0 else 1.0
        vals2 = np.prod(l - np.arange(k)) if k > 0 else 1.0
        log_c = (
            gammaln(k + 1)
            - gammaln(k - m + 1)
            + np.log(np.abs(vals2) + 1e-300)
            - gammaln(k + 1)
            + np.log(np.abs(val) + 1e-300)
            - gammaln(l + 1)
        )
        sign_c = np.sign(vals2) * np.sign(val)
        c = sign_c * np.exp(log_c)

        term = m * cos_t ** (k - m + 1) * sin_t ** (m - 1) - (k - m) * sin_t ** (
            m + 1
        ) * cos_t ** (k - m - 1)
        dpl = dpl + b * c * term

    # Handle NaN/Inf from 0^negative
    dpl[~np.isfinite(dpl)] = 0.0
    return dpl


def spherical_harmonics(theta, phi, L):
    """Compute real spherical harmonics and angular derivatives.

    Parameters
    ----------
    theta : ndarray, shape (n,)
        Colatitude in radians.
    phi : ndarray, shape (n,)
        Longitude in radians.
    L : int
        Maximum harmonic order.

    Returns
    -------
    slm : ndarray, shape (n, L^2+2L)
    dslm_dphi : ndarray, shape (n, L^2+2L)
    dslm_dtheta : ndarray, shape (n, L^2+2L)
    """
    n_cols = L**2 + 2 * L
    n_ch = len(theta)
    slm = np.zeros((n_ch, n_cols))
    dslm_dphi = np.zeros((n_ch, n_cols))
    dslm_dtheta = np.zeros((n_ch, n_cols))

    count = 0
    for l in range(1, L + 1):
        for m in range(-l, l + 1):
            am = abs(m)
            a = (-1) ** m * np.sqrt(
                (2 * l + 1)
                / (2 * np.pi)
                * np.exp(gammaln(l - am + 1) - gammaln(l + am + 1))
            )

            if m < 0:
                Lval = associated_legendre(np.cos(theta), l, am)
                slm[:, count] = a * Lval * np.sin(am * phi)
                dslm_dphi[:, count] = am * a * Lval * np.cos(am * phi)
                Ld = associated_legendre_deriv(theta, l, am)
                dslm_dtheta[:, count] = a * Ld * np.sin(am * phi)

            elif m == 0:
                Lval = associated_legendre(np.cos(theta), l, 0)
                norm = np.sqrt((2 * l + 1) / (4 * np.pi))
                slm[:, count] = norm * Lval
                dslm_dphi[:, count] = 0.0
                Ld = associated_legendre_deriv(theta, l, 0)
                dslm_dtheta[:, count] = norm * Ld

            else:  # m > 0
                Lval = associated_legendre(np.cos(theta), l, m)
                slm[:, count] = a * Lval * np.cos(m * phi)
                dslm_dphi[:, count] = (-m) * a * Lval * np.sin(m * phi)
                Ld = associated_legendre_deriv(theta, l, m)
                dslm_dtheta[:, count] = a * Ld * np.cos(m * phi)

            count += 1

    return slm, dslm_dphi, dslm_dtheta


def spheroid_fit(positions_m):
    """Fit a prolate spheroid to sensor positions.

    Parameters
    ----------
    positions_m : ndarray, shape (n, 3)
        Sensor positions in metres (MNE convention).

    Returns
    -------
    center : ndarray, shape (3,)
        Spheroid centre in metres.
    radii : ndarray, shape (3,)
        Semi-axis lengths in metres.
    longest_axis : int
        Index (0, 1, or 2) of the longest axis.
    """
    # Work in mm
    positions = positions_m * 1000.0

    vrange = np.abs(positions.max(axis=0) - positions.min(axis=0))
    longest_axis = int(np.argmax(vrange))

    center, radii = _spheroid_fit_axis(positions, longest_axis)

    # Convert back to metres
    return center / 1000.0, radii / 1000.0, longest_axis


def _spheroid_fit_axis(X, ax):
    """Core spheroid fit for a given longest axis.

    Direct translation of MATLAB spheroid_fit(X, ax).

    Parameters
    ----------
    X : ndarray, shape (n, 3)
        Sensor positions in mm.
    ax : int
        Index of the longest axis (0, 1, or 2).

    Returns
    -------
    o : ndarray, shape (3,)
        Centre in mm.
    r : ndarray, shape (3,)
        Radii in mm.
    """
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    b = x**2 + y**2 + z**2

    if ax == 0:
        A = np.column_stack(
            [
                y**2 + z**2 - 2 * x**2,
                2 * x,
                2 * y,
                2 * z,
                np.ones(len(x)),
            ]
        )
        beta = np.linalg.pinv(A) @ b
        v1 = -2 * beta[0] - 1
        v2 = beta[0] - 1
        v3 = beta[0] - 1
    elif ax == 1:
        A = np.column_stack(
            [
                x**2 + z**2 - 2 * y**2,
                2 * x,
                2 * y,
                2 * z,
                np.ones(len(x)),
            ]
        )
        beta = np.linalg.pinv(A) @ b
        v1 = beta[0] - 1
        v2 = -2 * beta[0] - 1
        v3 = beta[0] - 1
    elif ax == 2:
        A = np.column_stack(
            [
                x**2 + y**2 - 2 * z**2,
                2 * x,
                2 * y,
                2 * z,
                np.ones(len(x)),
            ]
        )
        beta = np.linalg.pinv(A) @ b
        v1 = beta[0] - 1
        v2 = beta[0] - 1
        v3 = -2 * beta[0] - 1
    else:
        raise ValueError(f"ax must be 0, 1, or 2, got {ax}")

    v = np.array([v1, v2, v3, 0.0, 0.0, 0.0, beta[1], beta[2], beta[3], beta[4]])

    # Build 4x4 matrix
    Amat = np.array(
        [
            [v[0], v[3], v[4], v[6]],
            [v[3], v[1], v[5], v[7]],
            [v[4], v[5], v[2], v[8]],
            [v[6], v[7], v[8], v[9]],
        ]
    )

    o = -np.linalg.solve(Amat[:3, :3], v[6:9])

    T = np.eye(4)
    T[3, :3] = o
    R = T @ Amat @ T.T

    s_vals, vec = np.linalg.eig(R[:3, :3] / (-R[3, 3]))
    r = np.sqrt(1.0 / np.abs(s_vals))
    sgns = np.sign(s_vals)
    r = r * sgns
    r = vec @ r

    return o, r


def shrink_spheroid(positions_m, center_m, radii_m):
    """Iteratively shrink radii by 1 mm until all sensors are outside.

    Parameters
    ----------
    positions_m : ndarray, shape (n, 3)
        Sensor positions in metres.
    center_m : ndarray, shape (3,)
        Spheroid centre in metres.
    radii_m : ndarray, shape (3,)
        Initial semi-axis lengths in metres.

    Returns
    -------
    radii_m : ndarray, shape (3,)
        Shrunk radii in metres.
    """
    # Work in mm
    v = (positions_m - center_m) * 1000.0
    r = radii_m.copy() * 1000.0

    inside = (
        v[:, 0] ** 2 / r[0] ** 2 + v[:, 1] ** 2 / r[1] ** 2 + v[:, 2] ** 2 / r[2] ** 2
    )
    c = np.sum(inside < 1)

    while c > 0:
        rt = r - 1.0
        inside = (
            v[:, 0] ** 2 / rt[0] ** 2
            + v[:, 1] ** 2 / rt[1] ** 2
            + v[:, 2] ** 2 / rt[2] ** 2
        )
        cc = np.sum(inside < 1)
        if cc <= c:
            r = r - 1.0
            c = cc

    return r / 1000.0


def cartesian_to_prolate(positions_m, orientations, center_m, a_m, b_m, longest_axis):
    """Convert Cartesian sensor coords to prolate spheroidal coords.

    The MATLAB code hardcodes column index 1 (0-based) as the major axis
    (Y in MATLAB = anterior). This function permutes columns so the
    dynamically-determined longest axis maps to that position.

    Parameters
    ----------
    positions_m : ndarray, shape (n, 3)
        Sensor positions in metres.
    orientations : ndarray, shape (n, 3)
        Sensor orientations (unit vectors).
    center_m : ndarray, shape (3,)
        Spheroid centre in metres.
    a_m : float
        Semi-major axis length in metres.
    b_m : float
        Semi-minor axis length in metres.
    longest_axis : int
        Index of the longest axis (0, 1, or 2).

    Returns
    -------
    major : ndarray, shape (n,)
    nabla : ndarray, shape (n,)
    phi : ndarray, shape (n,)
    emajor : ndarray, shape (n,)
        Projection of major unit vector onto sensor orientation.
    enabla : ndarray, shape (n,)
        Projection of nabla unit vector onto sensor orientation.
    ephi : ndarray, shape (n,)
        Projection of phi unit vector onto sensor orientation.
    hmajor : ndarray, shape (n,)
    hnabla : ndarray, shape (n,)
    hphi : ndarray, shape (n,)
    """
    # Work in mm
    v = (positions_m - center_m) * 1000.0
    a = a_m * 1000.0
    b = b_m * 1000.0
    n = orientations.copy()

    # Permute so longest axis is in column 1 (MATLAB Y position)
    # MATLAB code uses: col0 -> x (used in atan2 + cos(phi)),
    #                   col1 -> y (major axis),
    #                   col2 -> z (used in atan2 + sin(phi))
    if longest_axis == 0:
        # X is longest: map X->col1, Y->col0, Z->col2
        perm = [1, 0, 2]
    elif longest_axis == 1:
        # Y is longest: identity (matches MATLAB)
        perm = [0, 1, 2]
    elif longest_axis == 2:
        # Z is longest: map Z->col1, X->col0, Y->col2
        perm = [0, 2, 1]
    else:
        raise ValueError(f"longest_axis must be 0, 1, or 2, got {longest_axis}")

    v = v[:, perm]
    n = n[:, perm]

    # Focus
    c = np.sqrt(a**2 - b**2)
    T = np.sum(v**2, axis=1) + c**2

    # Prolate coordinates
    major = np.sqrt(T + np.sqrt(T**2 - 4 * v[:, 1] ** 2 * c**2)) / np.sqrt(2)
    phi = np.arctan2(v[:, 2], v[:, 0])

    tmp = v[:, 1] / major
    tmp = np.clip(tmp, -1, 1)
    nabla = np.arccos(tmp)

    # Unit vector projections onto sensor orientations
    denom = np.sqrt(major**2 - c**2 * np.cos(nabla) ** 2)
    minor = np.sqrt(major**2 - c**2)

    emajor = (
        major * np.sin(nabla) * np.cos(phi) * n[:, 0]
        + major * np.sin(nabla) * np.sin(phi) * n[:, 2]
        + minor * np.cos(nabla) * n[:, 1]
    )
    emajor /= denom

    enabla = (
        minor * np.cos(nabla) * np.cos(phi) * n[:, 0]
        + minor * np.cos(nabla) * np.sin(phi) * n[:, 2]
        - major * np.sin(nabla) * n[:, 1]
    )
    enabla /= denom

    ephi = np.cos(phi) * n[:, 2] - np.sin(phi) * n[:, 0]

    # Metric coefficients
    hmajor = np.sqrt((major**2 - c**2 * np.cos(nabla) ** 2) / (major**2 - c**2))
    hnabla = np.sqrt(major**2 - c**2 * np.cos(nabla) ** 2)
    hphi = np.sqrt(major**2 - c**2) * np.sin(nabla)

    return major, nabla, phi, emajor, enabla, ephi, hmajor, hnabla, hphi


def _compute_radial_functions(major, a, c, L, func):
    """Compute radial function for all (l, m) pairs.

    Parameters
    ----------
    major : ndarray, shape (n,)
    a, c : float
    L : int
    func : callable
        One of qlm_hat, dqlm_hat_dmajor, plm_hat, dplm_hat_dmajor.

    Returns
    -------
    result : ndarray, shape (n, L^2+2L)
    """
    n_cols = L**2 + 2 * L
    result = np.zeros((len(major), n_cols))
    count = 0
    for l in range(1, L + 1):
        for m in range(-l, l + 1):
            result[:, count] = func(major, a, c, l, m)
            count += 1
    return result


def qlm_hat(major, a, c, l, m):
    """Internal radial function via infinite series.

    Translated from spm_ipharm.m lines 84-113.

    Parameters
    ----------
    major : ndarray, shape (n,)
        Major coordinate values (mm).
    a : float
        Semi-major axis (mm).
    c : float
        Focus distance (mm).
    l : int
        Degree.
    m : int
        Order (uses abs(m)).

    Returns
    -------
    qlm : ndarray, shape (n,)
    """
    m = abs(m)

    if c == 0:
        lt1t2 = (-l - m - 1) * np.log(major / a) + m / 2 * np.log(
            (major**2 - c**2) / (a**2 - c**2 + 1e-32)
        )
        return np.exp(lt1t2)

    # k = 0
    lg = gammaln((1 + l + m) / 2) + gammaln((2 + l + m) / 2) - gammaln(l + 1.5)
    F1 = np.full_like(major, np.exp(lg))
    F2 = np.exp(lg)

    k = 0
    F1_tmp = np.exp(lg + 2 * k * np.log(c / major))
    F2_tmp = np.exp(lg + 2 * k * np.log(c / a))
    F1 = F1_tmp.copy()
    F2_scalar = F2_tmp

    check_max = max(np.max(F1_tmp), F2_tmp)

    k = 1
    while check_max > 1e-32 and k < 5000:
        lg = (
            gammaln((1 + l + m) / 2 + k)
            + gammaln((2 + l + m) / 2 + k)
            - gammaln(l + 1.5 + k)
            - np.sum(np.log(np.arange(1, k + 1)))
        )
        F1_tmp = np.exp(lg + 2 * k * np.log(c / major))
        F1 += F1_tmp
        F2_tmp = np.exp(lg + 2 * k * np.log(c / a))
        F2_scalar += F2_tmp
        check_max = max(np.max(F1_tmp), F2_tmp)
        k += 1

    lt1t2 = (-l - m - 1) * np.log(major / a) + m / 2 * np.log(
        (major**2 - c**2) / (a**2 - c**2)
    )
    qlm = np.exp(lt1t2 + np.log(F1) - np.log(F2_scalar))

    return qlm


def dqlm_hat_dmajor(major, a, c, l, m):
    """Derivative of internal radial function w.r.t. major.

    Translated from spm_ipharm.m lines 129-182.

    Parameters
    ----------
    major : ndarray, shape (n,)
    a, c : float
    l, m : int

    Returns
    -------
    dqlm : ndarray, shape (n,)
    """
    m = abs(m)
    c = c + 1e-32  # avoid log(0)

    # Compute F1, F2 (same series as qlm_hat but with 1e-16 convergence)
    k = 0
    lg = (
        gammaln((1 + l + m) / 2 + k)
        + gammaln((2 + l + m) / 2 + k)
        - gammaln(l + 1.5 + k)
    )
    F1 = np.exp(lg + 2 * k * np.log(c / major))
    F2_scalar = np.exp(lg + 2 * k * np.log(c / a))

    check_max = max(
        np.max(np.exp(lg + 2 * k * np.log(c / major))),
        np.exp(lg + 2 * k * np.log(c / a)),
    )

    k = 1
    while check_max > 1e-16 and k < 5000:
        lg = (
            gammaln((1 + l + m) / 2 + k)
            + gammaln((2 + l + m) / 2 + k)
            - gammaln(l + 1.5 + k)
            - np.sum(np.log(np.arange(1, k + 1)))
        )
        F1_tmp = np.exp(lg + 2 * k * np.log(c / major))
        F1 += F1_tmp
        F2_tmp = np.exp(lg + 2 * k * np.log(c / a))
        F2_scalar += F2_tmp
        check_max = max(np.max(F1_tmp), F2_tmp)
        k += 1

    lt1t2 = (-l - m - 1) * np.log(major / a) + m / 2 * np.log(
        (major**2 - c**2) / (a**2 - c**2)
    )
    u = np.exp(lt1t2)
    v = F1 / F2_scalar

    # In MATLAB, dvdmajor is always 0 (the loop never executes).
    # We replicate this behavior exactly.
    dvdmajor = np.zeros_like(major)

    # Compute dudmajor
    minor = np.sqrt(major**2 - c**2)

    lt1 = (
        (-l - m - 2) * np.log(major)
        + m * np.log(minor)
        - (-l - m - 1) * np.log(a)
        - m / 2 * np.log(a**2 - c**2)
    )
    lt2 = (
        np.log(m + 1e-32)
        + (-l - m) * np.log(major)
        + (m - 2) * np.log(minor)
        - (-l - m - 1) * np.log(a)
        - m / 2 * np.log(a**2 - c**2)
    )

    dudmajor = (-l - m - 1) * np.exp(lt1) + np.exp(lt2)

    dqlm = u * dvdmajor + v * dudmajor

    return dqlm


def plm_hat(major, a, c, l, m):
    """External radial function via finite series.

    Translated from spm_epharm.m lines 84-107.

    Parameters
    ----------
    major : ndarray, shape (n,)
        Major coordinate values (mm).
    a : float
        Semi-major axis (mm).
    c : float
        Focus distance (mm).
    l : int
        Degree.
    m : int
        Order (uses abs(m)).

    Returns
    -------
    plm : ndarray, shape (n,)
    """
    c = c + 1e-16  # avoid log(0)
    m = abs(m)

    # k = 0
    lg = gammaln(2 * l + 1) - gammaln(1) - gammaln(l + 1) - gammaln(l - m + 1)
    num = np.full_like(major, lg)
    denom = lg  # scalar

    for k in range(1, l // 2 + 1):
        if (l - 2 * k - m + 1) > 0:
            lg_k = (
                gammaln(2 * l - 2 * k + 1)
                - gammaln(k + 1)
                - gammaln(l - k + 1)
                - gammaln(l - 2 * k - m + 1)
            )
            lnumtmp = lg_k + 2 * k * np.log(c / major)
            ldenomtmp = lg_k + 2 * k * np.log(c / a)

            # Log-sum-exp with alternating signs
            num = num + np.log(1 + ((-1) ** k) * np.exp(lnumtmp - num))
            denom = denom + np.log(1 + ((-1) ** k) * np.exp(ldenomtmp - denom))

    series = num - denom
    lt1t2 = (l - m) * np.log(major / a) + m / 2 * np.log(
        (major**2 - c**2) / (a**2 - c**2)
    )
    plm = np.exp(lt1t2 + series)

    return plm


def dplm_hat_dmajor(major, a, c, l, m):
    """Derivative of external radial function w.r.t. major.

    Translated from spm_epharm.m lines 122-171.

    Parameters
    ----------
    major : ndarray, shape (n,)
    a, c : float
    l, m : int

    Returns
    -------
    dplm : ndarray, shape (n,)
    """
    c = c + 1e-16
    m = abs(m)

    # u = exp(lt1t2)
    lt1t2 = (l - m) * np.log(major / a) + m / 2 * np.log(
        (major**2 - c**2) / (a**2 - c**2)
    )
    u = np.exp(lt1t2)

    # dudr
    minor = np.sqrt(major**2 - c**2)
    minorref = np.sqrt(a**2 - c**2)
    dudr1 = (
        np.log(np.maximum(l - m, 1e-32))
        + (l - m - 1) * np.log(major)
        + m * np.log(minor)
        - (l - m) * np.log(a)
        - m * np.log(minorref)
    )
    dudr2 = (
        np.log(np.maximum(m, 1e-32))
        + (l - m + 1) * np.log(major)
        + (m - 2) * np.log(minor)
        - (l - m) * np.log(a)
        - m * np.log(minorref)
    )
    dudr = np.exp(dudr1) + np.exp(dudr2)

    # v and dvdr
    lg0 = gammaln(2 * l + 1) - gammaln(1) - gammaln(l + 1) - gammaln(l - m + 1)
    num = np.full_like(major, lg0)
    denom = lg0
    num2 = np.zeros_like(major)

    for k in range(1, l // 2 + 1):
        if (l - 2 * k - m + 1) > 0:
            lg_k = (
                gammaln(2 * l - 2 * k + 1)
                - gammaln(k + 1)
                - gammaln(l - k + 1)
                - gammaln(l - 2 * k - m + 1)
            )

            lnumtmp = lg_k + 2 * k * np.log(c / major)
            num = num + np.log(1 + ((-1) ** k) * np.exp(lnumtmp - num))

            ldenomtmp = lg_k + 2 * k * np.log(c / a)
            denom = denom + np.log(1 + ((-1) ** k) * np.exp(ldenomtmp - denom))

            lnumtmp2 = (
                (-1) ** k
                * np.exp(lg_k)
                * (-2 * k)
                * (c / major) ** (2 * k)
                * (1 / major)
            )
            num2 += lnumtmp2

    v = np.exp(num - denom)
    dvdr = num2 / np.exp(denom)

    dplm = v * dudr + u * dvdr

    return dplm


def _compute_harmonic_basis(
    positions_m,
    orientations,
    a_m,
    b_m,
    L,
    center_m,
    longest_axis,
    radial_func,
    radial_deriv_func,
):
    """Compute prolate spheroidal harmonic basis (internal or external).

    Parameters
    ----------
    positions_m : ndarray, shape (n, 3)
        Sensor positions in metres.
    orientations : ndarray, shape (n, 3)
        Sensor orientations.
    a_m : float
        Semi-major axis in metres.
    b_m : float
        Semi-minor axis in metres.
    L : int
        Harmonic order.
    center_m : ndarray, shape (3,)
        Spheroid centre in metres.
    longest_axis : int
        Index of the longest axis.
    radial_func : callable
        Radial function (qlm_hat or plm_hat).
    radial_deriv_func : callable
        Radial derivative (dqlm_hat_dmajor or dplm_hat_dmajor).

    Returns
    -------
    harmonics : ndarray, shape (n, L^2+2L)
        Normalized harmonic basis.
    """
    # cartesian_to_prolate works internally in mm; major and metric
    # coefficients are returned in mm.
    a = a_m * 1000.0
    b = b_m * 1000.0
    c = np.sqrt(a**2 - b**2)

    major, nabla, phi, emajor, enabla, ephi, hmajor, hnabla, hphi = (
        cartesian_to_prolate(
            positions_m, orientations, center_m, a_m, b_m, longest_axis
        )
    )
    # major, hmajor, hnabla, hphi are already in mm

    # Spherical harmonics
    slm, dslm_dphi, dslm_dnabla = spherical_harmonics(nabla, phi, L)

    # Radial functions (all in mm)
    rl = _compute_radial_functions(major, a, c, L, radial_func)
    drl = _compute_radial_functions(major, a, c, L, radial_deriv_func)

    # Prolate harmonic derivatives
    dpslm_dphi = rl * dslm_dphi
    dpslm_dnabla = rl * dslm_dnabla
    dpslm_dmajor = drl * slm

    # Gradient assembly
    Gphi = (ephi / hphi)[:, np.newaxis] * dpslm_dphi
    Gnabla = (enabla / hnabla)[:, np.newaxis] * dpslm_dnabla
    Gmajor = (emajor / hmajor)[:, np.newaxis] * dpslm_dmajor

    # Handle hphi == 0 (only needed for internal; harmless for external)
    Gphi[hphi == 0, :] = 0.0

    harmonics = Gmajor + Gphi + Gnabla

    # Normalize: zero-mean, unit-variance
    harmonics -= np.nanmean(harmonics, axis=0, keepdims=True)
    std = np.nanstd(harmonics, axis=0, keepdims=True, ddof=0)
    std[std == 0] = 1.0
    harmonics /= std

    return harmonics


def compute_internal_harmonics(
    positions_m, orientations, a_m, b_m, L, center_m, longest_axis
):
    """Compute internal prolate spheroidal harmonic basis.

    Parameters
    ----------
    positions_m : ndarray, shape (n, 3)
        Sensor positions in metres.
    orientations : ndarray, shape (n, 3)
        Sensor orientations.
    a_m : float
        Semi-major axis in metres.
    b_m : float
        Semi-minor axis in metres.
    L : int
        Harmonic order.
    center_m : ndarray, shape (3,)
        Spheroid centre in metres.
    longest_axis : int
        Index of the longest axis.

    Returns
    -------
    harmonics : ndarray, shape (n, L^2+2L)
        Normalized internal harmonic basis.
    """
    return _compute_harmonic_basis(
        positions_m,
        orientations,
        a_m,
        b_m,
        L,
        center_m,
        longest_axis,
        qlm_hat,
        dqlm_hat_dmajor,
    )


def compute_external_harmonics(
    positions_m, orientations, a_m, b_m, L, center_m, longest_axis
):
    """Compute external prolate spheroidal harmonic basis.

    Parameters
    ----------
    positions_m : ndarray, shape (n, 3)
        Sensor positions in metres.
    orientations : ndarray, shape (n, 3)
        Sensor orientations.
    a_m : float
        Semi-major axis in metres.
    b_m : float
        Semi-minor axis in metres.
    L : int
        Harmonic order.
    center_m : ndarray, shape (3,)
        Spheroid centre in metres.
    longest_axis : int
        Index of the longest axis.

    Returns
    -------
    harmonics : ndarray, shape (n, L^2+2L)
        Normalized external harmonic basis.
    """
    return _compute_harmonic_basis(
        positions_m,
        orientations,
        a_m,
        b_m,
        L,
        center_m,
        longest_axis,
        plm_hat,
        dplm_hat_dmajor,
    )


def _orth(A):
    """Orthonormal basis for column space.

    Uses SVD and keeps columns with singular values > max(size(A)) * eps * s[0].
    """
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    tol = max(A.shape) * np.finfo(float).eps * s[0]
    rank = int(np.sum(s > tol))
    return U[:, :rank]


def apply_amm(raw, li=9, le=2, window=10.0, corr_lim=1.0):
    """Apply AMM denoising.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw data with MEG channel positions and orientations.
    li : int
        Internal harmonic order. Default: 9.
    le : int
        External harmonic order. Default: 2.
    window : float
        Temporal window size in seconds. Default: 10.
    corr_lim : float
        CCA correlation limit (1.0 = no CCA). Default: 1.0.

    Returns
    -------
    raw_amm : mne.io.Raw
        Copy of raw with denoised data.
    info : dict
        Spheroid fitting info (center, radii, a, b, longest_axis).

    References
    ----------
    Tierney, T.M., Seedat, Z., St Pier, K. et al. (2024). Adaptive
    multipole models of optically pumped magnetometer data. Human Brain
    Mapping, 45, e26596. https://doi.org/10.1002/hbm.26596
    """
    raw_amm = raw.copy()

    # Get MEG channel indices (excluding bads)
    meg_picks = mne.pick_types(raw.info, meg=True, exclude="bads")
    meg_ch_names = [raw.ch_names[i] for i in meg_picks]

    # Extract positions and orientations from channel info
    n_ch = len(meg_picks)
    positions = np.zeros((n_ch, 3))
    orientations = np.zeros((n_ch, 3))

    for i, pick in enumerate(meg_picks):
        loc = raw.info["chs"][pick]["loc"]
        positions[i] = loc[:3]  # position in metres
        orientations[i] = loc[9:12]  # orientation (ez unit vector)

    # Check we have valid positions
    if np.all(positions == 0):
        raise ValueError("No sensor positions found in channel info.")

    print(
        f"  AMM: {n_ch} MEG channels, li={li}, le={le}, "
        f"window={window}s, corr_lim={corr_lim}"
    )

    # Fit spheroid
    center, radii, longest_axis = spheroid_fit(positions)
    print(f"  Spheroid centre: {center * 1000} mm")
    print(f"  Spheroid radii: {radii * 1000} mm")
    print(f"  Longest axis: {longest_axis} " f"({'XYZ'[longest_axis]})")

    # Centre positions
    positions_centered = positions - center

    # Shrink spheroid until all sensors are outside
    radii = shrink_spheroid(positions_centered, np.zeros(3), radii)
    print(f"  Shrunk radii: {radii * 1000} mm")

    a = np.max(np.abs(radii))
    b = np.min(np.abs(radii))
    print(f"  a={a * 1000:.1f} mm, b={b * 1000:.1f} mm")

    # Build harmonic bases
    print("  Computing external harmonics...")
    external = compute_external_harmonics(
        positions, orientations, a, b, le, center, longest_axis
    )
    print("  Computing internal harmonics...")
    internal = compute_internal_harmonics(
        positions, orientations, a, b, li, center, longest_axis
    )

    # Check for NaN/Inf
    if not np.all(np.isfinite(external)):
        n_bad = np.sum(~np.isfinite(external))
        print(
            f"  WARNING: {n_bad} non-finite values in external harmonics, "
            "replacing with 0"
        )
        external = np.nan_to_num(external, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.all(np.isfinite(internal)):
        n_bad = np.sum(~np.isfinite(internal))
        print(
            f"  WARNING: {n_bad} non-finite values in internal harmonics, "
            "replacing with 0"
        )
        internal = np.nan_to_num(internal, nan=0.0, posinf=0.0, neginf=0.0)

    # Build projectors using SVD for numerical stability.
    # pinv can inflate the rank of Pin when M_int has near-zero singular
    # values, causing Pin + Pout to span the full space and making the
    # CCA step remove all signal.
    print("  Building projectors...")
    Pout = external @ np.linalg.pinv(external)
    M = np.eye(n_ch) - Pout
    M_int = M @ internal

    # SVD-based projector: keep components with singular values well
    # above numerical noise
    U, s_mint, _ = np.linalg.svd(M_int, full_matrices=False)
    tol = max(M_int.shape) * np.finfo(float).eps * s_mint[0]
    rank = int(np.sum(s_mint > tol))
    U_r = U[:, :rank]
    Pin = U_r @ (U_r.T @ M)
    print(f"  M_int rank: {rank} (of {M_int.shape[1]} columns)")

    # Check orthogonality
    orth_check = np.linalg.norm(Pin @ external)
    print(f"  ||Pin @ ext|| = {orth_check:.2e}")

    # Window-based denoising
    data = raw_amm.get_data()
    sfreq = raw.info["sfreq"]
    n_samples = data.shape[1]
    win_samples = int(window * sfreq)

    chunks = list(range(0, n_samples, win_samples))
    if chunks[-1] < n_samples:
        chunks.append(n_samples)

    print(f"  Processing {len(chunks) - 1} windows...")
    for i in range(len(chunks) - 1):
        start = chunks[i]
        end = chunks[i + 1]

        Y = data[meg_picks, start:end]
        inner = Pin @ Y

        if corr_lim < 1:
            outer = Pout @ Y
            inter = Y - inner - outer

            # Skip CCA if residual is negligible (happens when
            # internal + external bases span nearly full sensor space)
            inter_rms = np.sqrt(np.mean(inter**2))
            y_rms = np.sqrt(np.mean(Y**2))
            if inter_rms > 1e-6 * y_rms:
                # _orth(): SVD-based, keeps columns with significant
                # singular values (> max(size) * eps * s[0])
                Oinner = _orth(inner.T)
                Ointer = _orth(inter.T)

                # CCA
                C = Oinner.T @ Ointer
                _, Sc, Zt = np.linalg.svd(C, full_matrices=False)
                noise = Ointer @ Zt.T

                # Remove components above correlation limit
                n_remove = int(np.sum(Sc > corr_lim))
                if n_remove > 0:
                    noisevec = noise[:, :n_remove]
                    Beta = noisevec.T @ inner.T
                    mod = noisevec @ Beta
                    inner = inner - mod.T

        data[meg_picks, start:end] = inner

    raw_amm._data[meg_picks, :] = data[meg_picks, :]
    print("  AMM complete.")

    info = {
        "center": center,
        "radii": radii,
        "a": a,
        "b": b,
        "longest_axis": longest_axis,
        "Pin": Pin,
        "external": external,
    }

    return raw_amm, info
