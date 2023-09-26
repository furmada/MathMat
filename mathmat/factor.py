"""MathMat Matrix Factorization Toolbox."""

from .mathmat import MatrixTypeException, Matrix, DiagonalMatrix, Identity
import numpy as np
from scipy.linalg import lu as scipy_lu
from scipy.sparse.linalg import svds as sparse_svd, \
    splu as sparse_lu
from scipy.sparse import csc_array


def cholesky(M: Matrix):
    """Compute the Cholesky factorization $M = L L^H$."""
    if not M.is_hermitian():
        raise MatrixTypeException("Cholesky factorization", "Hermitian")
    if not M.is_posdef():
        raise MatrixTypeException(
            "Cholesky factorization", "positive-definite")
    L = np.linalg.cholesky(M.entries)
    L = Matrix(csc_array(L))
    L.set("dense", Matrix(L))
    L.set("triangular_L", True)
    M.set("cholesky", L)
    M.set("plu", (Identity(M.nc), L, L.transpose()))
    return L


def diagonalization(M: Matrix):
    """Compute the eigenvalue decomposition $M = P D P^{-1}$. Densifies."""
    if not M.is_diagonalizable():
        raise MatrixTypeException("diagonalize", "diagonalizable")
    if M.is_sparse():
        M = M.to_dense()
    P = Matrix(M.eigenvectors())
    D = DiagonalMatrix(M.eigenvalues())

    M.set("diagonalization", (D, P))
    return D, P


def lu(M: Matrix, densify=False):
    """Compute the pivoted LU factorization $M = P L U$.
    If the matrix is sparse, returns an object that can be used to solve
    linear systems."""
    if not M.is_square():
        raise MatrixTypeException("LU factorization", "square")
    if M.is_sparse():
        if densify:
            M = M.to_dense()
        else:
            fct = sparse_lu(M.entries)
            M.set("lu_solver", fct.solve)
            return fct
    P, L, U = scipy_lu(M.entries)
    P = Matrix(P)
    L = Matrix(csc_array(L))
    L.set("dense", Matrix(L))
    L.set("triangular_L", True)
    U = Matrix(csc_array(U))
    U.set("dense", Matrix(U))
    U.set("triangular_R", True)

    P.set("unitary", True)
    P.set("inverse", lambda: U.conjugate().transpose())
    M.set("plu", (P, L, U))
    return P, L, U


def qr(M: Matrix):
    """Compute the QR factorization $M = QR$."""
    Q, R = np.linalg.qr(M.entries, mode="reduced")
    Q = Matrix(Q)
    R = Matrix(R)
    R.set("triangular_U", True)
    # Store known properties of Q and R
    if M.is_complex():
        Q._computed.update({
            "unitary": True,
            "inverse": lambda: Q.conjugate().transpose()
        })
    else:
        Q._computed.update({
            "orthogonal": True,
            "inverse": lambda: Q.transpose()
        })

    M.set("qr", (Q, R))
    return Q, R


def svd(M: Matrix, tolerance=1e-10, sp_k="max"):
    """Compute the SVD $M = U \\Sigma V^H$.

    Note that for sparse matrices, there is a limit set by sparse.svds
    as to how many singular values can be found! `max` is set to half
    of the minimum dimension"""
    if M.is_sparse():
        # Use sparse SVD
        U, sigmas, V = sparse_svd(
            M.entries,
            k=max(2, int(0.1*min(M.nr, M.nc)) if sp_k == "max" else sp_k),
            solver="lobpcg")
        order = np.flip(np.argsort(sigmas))
        U = U[:, order]
        sigmas = sigmas[order]
        V = V[order, :]
    else:
        # Use dense SVD
        U, sigmas, V = np.linalg.svd(M.entries)
    U = Matrix(U)
    if M.is_square():
        # Use a sparse diagonal
        S = DiagonalMatrix(sigmas)
    else:
        # Place the sigmas into the right shape
        vals = np.zeros(shape=M.size)
        vals[:len(sigmas), :len(sigmas)] = np.diag(sigmas)
        S = Matrix(vals)
    V = Matrix(V)
    # Store the known properties of U, S, and V.
    if M.is_complex():
        U._computed.update({
            "unitary": True,
            "inverse": lambda: U.conjugate().transpose()
        })
        V._computed.update({
            "unitary": True,
            "inverse": lambda: V.conjugate().transpose()
        })
    else:
        U._computed.update({
            "orthogonal": True,
            "inverse": lambda: U.transpose()
        })
        V._computed.update({
            "orthogonal": True,
            "inverse": lambda: V.transpose()
        })

    M.set("svd", (U, S, V))
    M.set("sigmas", sigmas)
    return U, S, V
