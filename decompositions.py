# Copyright 2019 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from scipy.linalg import block_diag, sqrtm, polar, schur
from functools import lru_cache

def sympmat(n):
    r""" Returns the symplectic matrix of order n
    Args:
        n (int): order
        hbar (float): the value of hbar used in the definition
            of the quadrature operators
    Returns:
        array: symplectic matrix
    """
    idnt = np.identity(n)
    zero = np.zeros((n, n))
    return np.block([[zero, idnt], [-idnt, zero]])


@lru_cache()
def changebasis(n):
  r"""Change of basis matrix between the two Gaussian representation orderings.
  This is the matrix necessary to transform covariances matrices written
  in the (x_1, ..., x_n, p_1, ..., p_n) to the (x_1, p_1, ..., x_n, p_n) ordering
  Args:
      n (int): number of modes
  Returns:
      array: :math:`2n \times 2n` matrix
  """
  m = np.zeros((2 * n, 2 * n))
  for i in range(n):
    m[2 * i, i] = 1
    m[2 * i + 1, i + n] = 1
  return m


def takagi(N, tol=1e-13, rounding=13):
    r"""Autonne-Takagi decomposition of a complex symmetric (not Hermitian!) matrix.

    Note that singular values of N are considered equal if they are equal after np.round(values, tol).

    See :cite:`cariolaro2016` and references therein for a derivation.

    Args:
        N (array[complex]): square, symmetric matrix N
        rounding (int): the number of decimal places to use when rounding the singular values of N
        tol (float): the tolerance used when checking if the input matrix is symmetric: :math:`|N-N^T| <` tol

    Returns:
        tuple[array, array]: (rl, U), where rl are the (rounded) singular values,
            and U is the Takagi unitary, such that :math:`N = U \diag(rl) U^T`.
    """
    (n, m) = N.shape
    if n != m:
        raise ValueError("The input matrix must be square")

    error = np.linalg.norm(N - N.T) / n
    if error >= tol:
        raise ValueError("The input matrix is not symmetric (error = %f)" % error)

    # If the matrix is real one can use its eigendecomposition
    if np.isrealobj(N) or np.all(N.imag == 0):
        l, U = np.linalg.eigh(N.real)
        phase = np.ones(l.size, dtype=np.complex128)
        phase[l < 0] = 1j
        Uc = U * phase[np.newaxis, :]
        l = np.abs(l) # Takagi eigenvalues
        permutation = np.argsort(l)[::-1]
        # Rearrange the unitary and values so that they are decreasingly ordered
        l = l[permutation]
        Uc = Uc[:, permutation]
        return l, Uc

    v, l, ws = np.linalg.svd(N)

    roundedl = np.round(l, rounding)

    # Check for degeneracies (sorted in decreasing order)
    subspace = np.round(roundedl, rounding)
    degeneracies = np.unique(subspace, return_counts=True)[1][::-1]
    uniques = degeneracies.size

    w = ws.T.conj()

    # Generate lists with the degenerate column subspaces
    vas, was = [None] * uniques, [None] * uniques
    first = last = 0
    for i, degen in enumerate(degeneracies):
        last += degen
        vas[i] = v[:, first:last]
        was[i] = w[:, first:last]
        first = last

    # Generate the matrices qs of the degenerate subspaces
    qs = [None] * uniques
    for i in range(uniques):
        qs[i] = sqrtm(vas[i].T @ was[i])

    # Construct the Takagi unitary
    qb = block_diag(*qs)

    U = v @ np.conj(qb)
    return roundedl, U


def graph_embed(A, max_mean_photon=1.0, make_traceless=True, tol=1e-6):
    r"""Embed a graph into a Gaussian state.

    Given a graph in terms of a symmetric adjacency matrix
    (in general with arbitrary complex off-diagonal and real diagonal entries),
    returns the squeezing parameters and interferometer necessary for
    creating the Gaussian state whose off-diagonal parts are proportional to that matrix.

    Uses :func:`takagi`.

    Args:
        A (array[complex]): square, symmetric (weighted) adjacency matrix of the graph
        max_mean_photon (float): Threshold value. It guarantees that the mode with
            the largest squeezing has ``max_mean_photon`` as the mean photon number
            i.e., :math:`sinh(r_{max})^2 ==` ``max_mean_photon``.
        make_traceless (bool): Removes the trace of the input matrix, by performing the transformation
            :math:`\tilde{A} = A-\mathrm{tr}(A) \I/n`. This may reduce the amount of squeezing needed to encode
            the graph.
        tol (float): tolerance used when checking if the input matrix is symmetric: :math:`|A-A^T| <` tol

    Returns:
        tuple[array, array]: squeezing parameters of the input
            state to the interferometer, and the unitary matrix representing the interferometer
    """
    (m, n) = A.shape

    if m != n:
        raise ValueError("The matrix is not square.")

    if np.linalg.norm(A - A.T) / n >= tol:
        raise ValueError("The matrix is not symmetric.")

    if make_traceless:
        A = A - np.trace(A) * np.identity(n) / n

    s, U = takagi(A, tol=tol)
    sc = np.sqrt(1.0 + 1.0 / max_mean_photon)
    vals = -np.arctanh(s / (s[0] * sc))
    return vals, U


def williamson(V, tol=1e-11):
    r"""Williamson decomposition of positive-definite (real) symmetric matrix.

    See :ref:`williamson`.

    Note that it is assumed that the symplectic form is

    .. math:: \Omega = \begin{bmatrix}0&I\\-I&0\end{bmatrix}

    where :math:`I` is the identity matrix and :math:`0` is the zero matrix.

    See https://math.stackexchange.com/questions/1171842/finding-the-symplectic-matrix-in-williamsons-theorem/2682630#2682630

    Args:
        V (array[float]): positive definite symmetric (real) matrix
        tol (float): the tolerance used when checking if the matrix is symmetric: :math:`|V-V^T| \leq` tol

    Returns:
        tuple[array,array]: ``(Db, S)`` where ``Db`` is a diagonal matrix
            and ``S`` is a symplectic matrix such that :math:`V = S Db S^T`
    """
    (n, m) = V.shape

    if n != m:
        raise ValueError("The input matrix is not square")

    if n % 2 != 0:
      raise ValueError("The input matrix must have an even number of rows/columns")

    error = np.linalg.norm(V - V.T) / n
    if error >= tol:
        raise ValueError("The input matrix is not symmetric (error = %f)" % error)

    n = n // 2
    omega = sympmat(n)
    rotmat = changebasis(n)
    vals = np.linalg.eigvalsh(V)

    if np.any(vals <= 0):
        raise ValueError("Input matrix is not positive definite")

    Mm12 = sqrtm(np.linalg.inv(V)).real
    r1 = Mm12 @ omega @ Mm12
    s1, K = schur(r1)
    X = np.array([[0, 1], [1, 0]])
    I = np.identity(2)

    # In what follows I construct a permutation matrix p  so that the Schur matrix has
    # only positive elements above the diagonal
    # Also the Schur matrix uses the x_1, p_1, ..., x_n, p_n  ordering thus I use rotmat to
    # go to the ordering x_1, ..., x_n, p_1, ..., p_n

    seq = [None] * n
    for i in range(n):
        if s1[2*i, 2*i + 1] > 0:
            seq[i] = I
        else:
            seq[i] = X

    p = block_diag(*seq)
    Kt = K @ p
    s1t = p @ s1 @ p
    dd = rotmat.T @ s1t @ rotmat
    Ktt = Kt @ rotmat
    diag = 1 / np.diag(dd[:n, n:])
    Db = np.diag(np.tile(diag, 2))
    S = Mm12 @ Ktt @ sqrtm(Db)
    return Db, np.linalg.inv(S).T


def bloch_messiah(S, tol=1e-10, rounding=9):
    r"""Bloch-Messiah decomposition of a symplectic matrix.

    See :ref:`bloch_messiah`.

    Decomposes a symplectic matrix into two symplectic unitaries and squeezing transformation.
    It automatically sorts the squeezers so that they respect the canonical symplectic form.

    Note that it is assumed that the symplectic form is

    .. math:: \Omega = \begin{bmatrix}0&I\\-I&0\end{bmatrix}

    where :math:`I` is the identity matrix and :math:`0` is the zero matrix.

    As in the Takagi decomposition, the singular values of N are considered
    equal if they are equal after np.round(values, rounding).

    If S is a passive transformation, then return the S as the first passive
    transformation, and set the the squeezing and second unitary matrices to
    identity. This choice is not unique.

    For more info see:
    https://math.stackexchange.com/questions/1886038/finding-euler-decomposition-of-a-symplectic-matrix

    Args:
        S (array[float]): symplectic matrix
        tol (float): the tolerance used when checking if the matrix is symplectic:
            :math:`|S^T\Omega S-\Omega| \leq tol`
        rounding (int): the number of decimal places to use when rounding the singular values

    Returns:
        tuple[array]: Returns the tuple ``(ut1, st1, vt1)``. ``ut1`` and ``vt1`` are symplectic unitaries,
            and ``st1`` is diagonal and of the form :math:`= \text{diag}(s1,\dots,s_n, 1/s_1,\dots,1/s_n)`
            such that :math:`S = ut1  st1  v1`
    """
    (n, m) = S.shape

    if n != m:
        raise ValueError("The input matrix is not square")
    if n % 2 != 0:
        raise ValueError("The input matrix must have an even number of rows/columns")

    n = n // 2
    omega = sympmat(n)
    error = np.linalg.norm(S.T @ omega @ S - omega) / n
    if error >= tol:
        raise ValueError("The input matrix is not symplectic (error = %f)" % error)

    if np.linalg.norm(S.T @ S - np.eye(2 * n)) >= tol:

        u, sigma = polar(S, side='left')
        ss, uss = takagi(sigma, tol=tol, rounding=rounding)

        # Apply a permutation matrix so that the squeezers appear in the order
        # s_1, ..., s_n, 1/s_1, ..., 1/s_n
        pmat = np.block([[np.identity(n), np.zeros((n, n))],
                         [np.zeros((n, n)), np.identity(n)[::-1]]])
        ut = uss @ pmat

        # Apply a second permutation matrix to permute s
        # (and their corresponding inverses) to get the canonical symplectic form
        qomega = ut.T @ omega @ ut

        # Identify degenerate subspaces (values sorted in decreasing order)
        subspace = np.round(ss[:n], rounding)
        degeneracies = np.unique(subspace, return_counts=True)[1][::-1]
        uniques = degeneracies.size

        # Rotation matrices (not permutations) based on svd.
        # See Appendix B2 of Serafini's book for more details.
        u_list, v_list = [None] * uniques, [None] * uniques

        if uniques < n:
            first = last = 0
            for i, degen in enumerate(degeneracies):
                last += degen
                if degen > 1:
                    u_svd, _, v_svd = np.linalg.svd(qomega[first:last, n+first:n+last].real)
                    u_list[i] = u_svd
                    v_list[i] = v_svd.T
                else:
                    u_list[i] = np.sign(qomega[first, n+first])
                    v_list[i] = 1

                first = last

            pmat1 = block_diag(*(u_list + v_list))

        else:
            pmat1 = np.block([[np.diag(np.sign(qomega[:n, n:].real.diagonal())), np.zeros((n, n))],
                              [np.zeros((n, n)), np.eye(n)]])

        permutation = pmat @ pmat1
        st1 = permutation.T @ np.diag(ss) @ permutation
        ut1 = uss @ permutation
        v1 = ut1.T @ u

    else:
        ut1 = S
        st1 = np.eye(2 * n)
        v1 = np.eye(2 * n)

    return ut1.real, st1.real, v1.real
