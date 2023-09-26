"""MathMat Linear System Solving Toolbox"""

import lazy_import

from .mathmat import Matrix, Vector, MatrixSizeException, MatrixTypeException
from .factor import lu as lu_factor
import numpy as np
from scipy.linalg import solve_triangular

mathmat_random = lazy_import.lazy_module("mathmat.random")


class Krylov:
    """A collection of methods for generating Krylov subspaces."""

    @staticmethod
    def arnoldi_mgs(A: Matrix, b: Vector, k: int,
                    tolerance=1e-4, **_):
        """Generate $K_k = \\text{span} ( b, Ab, ... A^{k-1}b )$.
        Uses Modified Gram-Schmidt (non-repeated) full orthogonalization.
        Optionally configure the tolerance for stopping early."""
        # If rank is known, restrict max iteration
        k = min(k, A._computed.get("rank", k))
        # Check if we need to use conjugates
        use_conj = A.is_complex()

        def dot(q, v): return np.dot(np.conjugate(
            q), v) if use_conj else np.dot(q, v)
        # The Krylov basis vectors.
        K_k = np.zeros((len(b), k),
                       dtype=np._complex if use_conj else np.double)
        # Set the first vector to normalized b
        K_k[:, 0] = b.unit().entries.ravel()
        # The upper Hessenberg matrix
        H = np.zeros((k+1, k),
                     dtype=np._complex if use_conj else np.double)

        # Outer iteration
        for j in range(k-1):
            # Arnoldi step
            v = A.entries @ K_k[:, j]

            # Orthogonalization
            for i in range(j+1):
                # Get the Krylov vector to orthogonalize against
                q_i = K_k[:, i]
                # Compute the Hessenberg term
                H[i, j] = dot(q_i, v)
                # Orthogonalize
                v -= H[i, j] * q_i

            # Fill in the sub-diagonal entry
            H[j+1, j] = np.linalg.norm(v, ord=2)

            if j > 0 and H[j+1, j] <= tolerance:
                # Exact solution reached, trim the generated subspace
                K_k = K_k[:, :j-1]
                H = H[:j, :j-1]
                # We have learned the rank of A
                A.set("rank", j-1)
                break

            # Normalize and store the vector
            v /= H[j+1, j]
            K_k[:, j+1] = v

        K_k = Matrix(K_k)
        K_k.set("rank", K_k.entries.shape[1])
        K_k.set("unitary" if use_conj else "orthogonal", True)

        H = Matrix(H)
        H.set("upper_hessenberg", True)

        return K_k, H

    @staticmethod
    def arnoldi_mgs_trunc(A: Matrix, b: Vector, k: int,
                          tolerance=1e-4, r=10):
        """Generate $K_k = \\text{span} ( b, Ab, ... A^{k-1}b )$.
        Uses Modified Gram-Schmidt (non-repeated) `r`-truncated
        orthogonalization (see Nakatuskasa and Tropp 2022).
        Optionally configure the tolerance for stopping early."""
        # If rank is known, restrict max iteration
        k = min(k, A._computed.get("rank", k))
        # Check if we need to use conjugates
        use_conj = A.is_complex()

        def dot(q, v): return np.dot(np.conjugate(
            q), v) if use_conj else np.dot(q, v)
        # The Krylov basis vectors.
        K_k = np.zeros((len(b), k),
                       dtype=np._complex if use_conj else np.double)
        # Set the first vector to normalized b
        K_k[:, 0] = b.unit().entries.ravel()
        # The upper Hessenberg matrix
        H = np.zeros((k+1, k),
                     dtype=np._complex if use_conj else np.double)

        # Outer iteration
        for j in range(k-1):
            # Arnoldi step
            v = A.entries @ K_k[:, j]

            # Truncated Orthogonalization
            for i in range(max(0, j-r), j+1):
                # Get the Krylov vector to orthogonalize against
                q_i = K_k[:, i]
                # Compute the Hessenberg term
                H[i, j] = dot(q_i, v)
                # Orthogonalize
                v -= H[i, j] * q_i

            # Fill in the sub-diagonal entry
            H[j+1, j] = np.linalg.norm(v, ord=2)

            if j > 0 and H[j+1, j] <= tolerance:
                # Exact solution reached, trim the generated subspace
                K_k = K_k[:, :j-1]
                H = H[:j, :j-1]
                # We have learned the rank of A
                A.set("rank", j-1)
                break

            # Normalize and store the vector
            v /= H[j+1, j]
            K_k[:, j+1] = v

        K_k = Matrix(K_k)
        H = Matrix(H)
        H.set("upper_hessenberg", True)

        return K_k, H

    @staticmethod
    def arnoldi_mgs_2x(A: Matrix, b: Vector, k: int,
                       tolerance=1e-4, **_):
        """Generate $K_k = \\text{span} ( b, Ab, ... A^{k-1}b )$.
        Uses Modified Gram-Schmidt (twice) full orthogonalization.
        Optionally configure the tolerance for stopping early."""
        # If rank is known, restrict max iteration
        k = min(k, A._computed.get("rank", k))
        # Check if we need to use conjugates
        use_conj = A.is_complex()

        def dot(q, v): return np.dot(np.conjugate(
            q), v) if use_conj else np.dot(q, v)
        # The Krylov basis vectors.
        K_k = np.zeros((len(b), k),
                       dtype=np._complex if use_conj else np.double)
        # Set the first vector to normalized b
        K_k[:, 0] = b.unit().entries.ravel()
        # The upper Hessenberg matrix
        H = np.zeros((k+1, k),
                     dtype=np._complex if use_conj else np.double)

        # Outer iteration
        for j in range(k-1):
            # Arnoldi step
            v = A.entries @ K_k[:, j]

            # Orthogonalization
            for i in range(j+1):
                # Get the Krylov vector to orthogonalize against
                q_i = K_k[:, i]
                # Compute the Hessenberg term
                H[i, j] = dot(q_i, v)
                # Orthogonalize once
                v -= H[i, j] * q_i
                # Orthogonalize again
                h_tmp = dot(q_i, v)
                H[i, j] += h_tmp
                v -= h_tmp * q_i

            # Fill in the sub-diagonal entry
            H[j+1, j] = np.linalg.norm(v, ord=2)

            if j > 0 and H[j+1, j] <= tolerance:
                # Exact solution reached, trim the generated subspace
                K_k = K_k[:, :j-1]
                H = H[:j, :j-1]
                # We have learned the rank of A
                A.set("rank", j-1)
                break

            # Normalize and store the vector
            v /= H[j+1, j]
            K_k[:, j+1] = v

        K_k = Matrix(K_k)
        K_k.set("rank", K_k.entries.shape[1])
        K_k.set("unitary" if use_conj else "orthogonal", True)

        H = Matrix(H)
        H.set("upper_hessenberg", True)

        return K_k, H


def gmres(A: Matrix, b: Vector, k: int,
          krylov="auto", tolerance=1e-4):
    """Solve Ax=b for x iteratively using the GMRES algorithm.
    Stops after `k` iterations, or when the sub-diagonal entry
    is below the given `tolerance`.

    Optionally specify the method of generating the Krylov
    subspace by passing a callable to `krylov`. Otherwise,
    this is derermined automatically from A. The callable
    should accept the arguments `(A, b, k, tol)` and return the
    Matrices $K_k$ and $H$.
    """
    if krylov == "auto":
        krylov = Krylov.arnoldi_mgs

    # Use the specified method to compute the basis
    K_k, H = krylov(A, b, k, tolerance)

    # Solve the least-squares problem
    rhs = np.zeros((H.nr, 1))
    rhs[0] = b.norm_2()

    y = lstsq(H, Vector(rhs))
    x = K_k @ y

    return x, K_k, H


def sgmres(A: Matrix, b: Vector, k: int, r: int,
           krylov="auto", tolerance=1e-4):
    """Solve Ax=b for x iteratively using the sGMRES algorithm.
    Generates a Krylov subspace of dimension `k` via a truncated
    Arnoldi process that orthogonalizes against `r` previous vectors.
    The solution is then produced by sketching the matrix using a FFT.

    Optionally specify the method of generating the Krylov
    subspace by passing a callable to `krylov`. Otherwise,
    truncated MGS Arnoldi will be used. The callable
    should accept the arguments `(A, b, k, tol, r)` and return the
    Matrices $K_k$ and $H$.
    """
    if krylov == "auto":
        krylov = Krylov.arnoldi_mgs_trunc

    # Use the specified method to compute the basis
    K_k, H = krylov(A, b, k, tolerance, r)

    # Sketch
    SA = mathmat_random.FFTSketchedMatrix(A @ K_k)
    Sb = mathmat_random.FFTSketchedMatrix(b)
    rows = np.random.choice(len(b), min(len(b), 2*K_k.nc))
    SA = SA.entries[rows, :]
    Sb = Sb.entries[rows, :]

    x = np.linalg.lstsq(SA, Sb, rcond=None)[0]
    x = K_k.entries @ x

    return Vector(x), K_k, H


def invert(A: Matrix, b: Vector):
    """Solve $Ax=b$ for $x$ by inverting the matrix $A$."""
    if A.nr != b.nr:
        raise MatrixSizeException(A, b, "Linear System Solve")
    return A.inverse() @ b


def lstsq(A: Matrix, b: Vector):
    """Find the least-squares, min. norm solution to $Ax=b$."""
    if A.nr != b.nr:
        raise MatrixSizeException(A, b, "Least Squares Solve")
    if "qr" in A._computed:
        # Use the QR of A if it exists
        Q, R = A.qr()
        return R.lin_solver()(Q.transpose() @ b)
    # Computing lstsq also reveals some properties of $A$.
    x, _, rank, sigmas = np.linalg.lstsq(A.entries, b.entries, rcond=None)
    if "rank" not in A._computed:
        A.set("rank", rank)
    if "sigmas" not in A._computed:
        A.set("sigmas", np.flip(np.sort(sigmas)))
    return Vector(x)


def lu(A: Matrix, b: Vector):
    """Solve $Ax=b$ for $x% by taking an $LU$ factorization of $A$."""
    if not A.is_square():
        raise MatrixTypeException("LU factorization", "square")
    if A.is_sparse():
        # We obtain the sparse LU solver
        A.set("lin_solver", lambda v: Vector(lu_factor(A).solve(v.entries)))
        return Vector(A.lin_solver()(b))
    else:
        P, L, U = lu_factor(A)
        y = L.lin_solver()(P.inverse() @ b)
        return U.lin_solver()(y)


def triangular(A: Matrix, b: Vector, tolerance=1e-10):
    """Solve $Ax=b$ for $x$ for a triangular $A$."""
    if not (A.is_triangular_U(tolerance) or A.is_triangular_L(tolerance)):
        raise MatrixTypeException("triangular solve", "triangular (any)")
    return Vector(solve_triangular(A.entries, b.entries, check_finite=False))


def automatic(A: Matrix, b: Vector, get_method=False):
    """Try to pick an optimal solving strategy based on the properties of $A$.

    Note that computing some properties may be expensive, and therefore using
    `automatic` may be sub-optimal if used only once.
    Optionally, if `get_method` is `True`, return the solver method instead of
    actually solving for $x$.
    """
    if "lin_solver" not in A._computed:
        # We need to find a suitable linear system solver
        if "inverse" in A._computed:
            # An inverse exists and thus a direct solution is available.
            A.set("lin_solver", lambda v: invert(A, v))
            A.lin_solver().name = "auto: invert"
        elif A.is_triangular_L() or A.is_triangular_U():
            # A triangular matrix can be solved using a faster algorithm.
            A.set("lin_solver", lambda v: triangular(A, v))
            A.lin_solver().name = "auto: triangular"
        elif "qr" in A._computed:
            # If a QR exists then lstsq will use it
            A.set("lin_solver", lambda v: lstsq(A, v))
            A.lin_solver().name = "auto: qr"
        elif "plu" in A._computed or A.is_square():
            # If a LU (or Cholesky) exists or can be found, use the LU solver
            A.set("lin_solver", lambda v: lu(A, v))
            A.lin_solver().name = "auto: lu"
        else:
            # The matrix is nonsquare and has no available factorizations.
            A.set("lin_solver", lambda v: gmres(A, v, A.nc)[0])
            A.lin_solver().name = "auto: gmres"

    if get_method:
        return A._computed["lin_solver"]
    return A.lin_solver(b)
