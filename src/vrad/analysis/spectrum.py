"""Source code from the spectrum package.

lO DO: Remove once the spectrum package can be installed as a dependency.
"""

import numpy as np


def LEVINSON(r, order=None, allow_singularity=False):
    # from np import isrealobj
    T0 = np.real(r[0])
    T = r[1:]
    M = len(T)

    if order is None:
        M = len(T)
    else:
        assert order <= M, "order must be less than size of the input data"
        M = order

    realdata = np.isrealobj(r)
    if realdata is True:
        A = np.zeros(M, dtype=float)
        ref = np.zeros(M, dtype=float)
    else:
        A = np.zeros(M, dtype=complex)
        ref = np.zeros(M, dtype=complex)

    P = T0

    for k in range(0, M):
        save = T[k]
        if k == 0:
            temp = -save / P
        else:
            # save += sum([A[j]*T[k-j-1] for j in range(0,k)])
            for j in range(0, k):
                save = save + A[j] * T[k - j - 1]
            temp = -save / P
        if realdata:
            P = P * (1.0 - temp ** 2.0)
        else:
            P = P * (1.0 - (temp.real ** 2 + temp.imag ** 2))
        if P <= 0 and allow_singularity == False:
            raise ValueError("singular matrix")
        A[k] = temp
        ref[k] = temp  # save reflection coeff at each step
        if k == 0:
            continue

        khalf = (k + 1) // 2
        if realdata is True:
            for j in range(0, khalf):
                kj = k - j - 1
                save = A[j]
                A[j] = save + temp * A[kj]
                if j != kj:
                    A[kj] += temp * save
        else:
            for j in range(0, khalf):
                kj = k - j - 1
                save = A[j]
                A[j] = save + temp * A[kj].conjugate()
                if j != kj:
                    A[kj] = A[kj] + temp * save.conjugate()

    return A, P, ref


def rlevinson(a, efinal):
    a = np.array(a)
    realdata = np.isrealobj(a)

    assert a[0] == 1, "First coefficient of the prediction polynomial must be unity"

    p = len(a)

    if p < 2:
        raise ValueError("Polynomial should have at least two coefficients")

    if realdata == True:
        U = np.zeros((p, p))  # This matrix will have the prediction
        # polynomials of orders 1:p
    else:
        U = np.zeros((p, p), dtype=complex)
    U[:, p - 1] = np.conj(a[-1::-1])  # Prediction coefficients of order p

    p = p - 1
    e = np.zeros(p)

    # First we find the prediction coefficients of smaller orders and form the
    # Matrix U

    # Initialize the step down

    e[-1] = efinal  # Prediction error of order p

    # Step down
    for k in range(p - 1, 0, -1):
        [a, e[k - 1]] = levdown(a, e[k])
        U[:, k] = np.concatenate((np.conj(a[-1::-1].transpose()), [0] * (p - k)))

    e0 = e[0] / (1.0 - abs(a[1] ** 2))  #% Because a[1]=1 (true polynomial)
    U[0, 0] = 1  #% Prediction coefficient of zeroth order
    kr = np.conj(U[0, 1:])  #% The reflection coefficients
    kr = kr.transpose()  #% To make it into a column vector

    #   % Once we have the matrix U and the prediction error at various orders, we can
    #  % use this information to find the autocorrelation coefficients.

    R = np.zeros(1, dtype=complex)
    #% Initialize recursion
    k = 1
    R0 = e0  # To take care of the zero indexing problem
    R[0] = -np.conj(U[0, 1]) * R0  # R[1]=-a1[1]*R[0]

    # Actual recursion
    for k in range(1, p):
        r = -sum(np.conj(U[k - 1 :: -1, k]) * R[-1::-1]) - kr[k] * e[k - 1]
        R = np.insert(R, len(R), r)

    # Include R(0) and make it a column vector. Note the dot transpose

    # R = [R0 R].';
    R = np.insert(R, 0, e0)
    return R, U, kr, e


def levdown(anxt, enxt=None):
    #% Some preliminaries first
    # if nargout>=2 & nargin<2
    #    raise ValueError('Insufficient number of input arguments');
    if anxt[0] != 1:
        raise ValueError("At least one of the reflection coefficients is equal to one.")
    anxt = anxt[1:]  #  Drop the leading 1, it is not needed
    #  in the step down

    # Extract the k+1'th reflection coefficient
    knxt = anxt[-1]
    if knxt == 1.0:
        raise ValueError("At least one of the reflection coefficients is equal to one.")

    # A Matrix formulation from Stoica is used to avoid looping
    acur = (anxt[0:-1] - knxt * np.conj(anxt[-2::-1])) / (1.0 - abs(knxt) ** 2)
    ecur = None
    if enxt is not None:
        ecur = enxt / (1.0 - np.dot(knxt.conj().transpose(), knxt))

    acur = np.insert(acur, 0, 1)

    return acur, ecur


def levup(acur, knxt, ecur=None):
    if acur[0] != 1:
        raise ValueError("At least one of the reflection coefficients is equal to one.")
    acur = acur[1:]  #  Drop the leading 1, it is not needed

    # Matrix formulation from Stoica is used to avoid looping
    anxt = np.concatenate((acur, [0])) + knxt * np.concatenate(
        (np.conj(acur[-1::-1]), [1])
    )

    enxt = None
    if ecur is not None:
        # matlab version enxt = (1-knxt'.*knxt)*ecur
        enxt = (1.0 - np.dot(np.conj(knxt), knxt)) * ecur

    anxt = np.insert(anxt, 0, 1)

    return anxt, enxt


def arma2psd(A=None, B=None, rho=1.0, T=1.0, NFFT=4096):
    if NFFT is None:
        NFFT = 4096

    if A is None and B is None:
        raise ValueError("Either AR or MA model must be provided")

    psd = np.zeros(NFFT, dtype=complex)

    if A is not None:
        ip = len(A)
        den = np.zeros(NFFT, dtype=complex)
        den[0] = 1.0 + 0j
        for k in range(0, ip):
            den[k + 1] = A[k]
        denf = np.fft.fft(den, NFFT)

    if B is not None:
        iq = len(B)
        num = np.zeros(NFFT, dtype=complex)
        num[0] = 1.0 + 0j
        for k in range(0, iq):
            num[k + 1] = B[k]
        numf = np.fft.fft(num, NFFT)

    # Changed in version 0.6.9 (divided by T instead of multiply)
    if A is not None and B is not None:
        psd = rho / T * abs(numf) ** 2.0 / abs(denf) ** 2.0
    elif A is not None:
        psd = rho / T / abs(denf) ** 2.0
    elif B is not None:
        psd = rho / T * abs(numf) ** 2.0

    psd = np.real(psd)

    return psd
