"""MathMat is a mathematically aware matrix library for Python.

See the Matrix class for more details.
"""

import lazy_import
from functools import wraps
from sys import float_info
from scipy.sparse.linalg import eigs as sparse_eigs, eigsh as sparse_eigsh, \
    norm as sparse_norm
from scipy.sparse import identity, issparse, dia_array, csc_array, \
    tril as sparse_tril, triu as sparse_triu
from scipy.linalg import issymmetric, ishermitian, eig, eigh, eigvals, \
    eigvalsh, inv
import numpy as np


factor = lazy_import.lazy_module("mathmat.factor")
solve = lazy_import.lazy_module("mathmat.solve")

EPS = float_info.epsilon


def uniquetol(arr, tolerance):
    """Find and return the unique elements of `arr` within a `tolerance`."""
    unique = np.unique(arr)
    if len(unique) <= 1:
        return unique
    u_diff = np.abs(np.diff(unique))
    where = np.argwhere(u_diff > tolerance)
    return np.insert(unique[where + 1], 0, unique[0])


class MatrixSizeException(Exception):
    """A `MatrixSizeException` is raised when two matrices have
    incompatible sizes for an operation, usually multiplication."""

    def __init__(self, A, B, operation):
        super().__init__(
            "Incompatible matrix sizes {}x{} and {}x{} for {}.".format(
                A.nr, A.nc, B.nr, B.nc, operation))


class MatrixTypeException(Exception):
    """A `MatrixTypeException` is raised when a matrix does not satisfy
    the requirements of an attempted operation."""

    def __init__(self, operation, req_type):
        super().__init__(
            "The matrix must be {} to perform: {}.".format(
                req_type, operation))


def _computable_property(name):
    """Decorate a function as defining a Computable Property of a Matrix."""
    def decorator(func):
        @wraps(func)
        def property_wrapper(*args, **kwargs):
            """Check if the Property already has a value."""
            if name not in args[0]._computed:
                # No value exists, perform the computation.
                args[0].set(name, func(*args, **kwargs))
            value = args[0]._computed.get(name, None)
            if callable(value) and not name == "lin_solver":
                # A value can be generated using another function.
                value = value()
                args[0].set(name, value)
            return value
        return property_wrapper
    return decorator


class Matrix:
    """A Matrix object is an immutable representation of a matrix.

    In addition to the entries, the Matrix keeps track of
    computations which have been performed on it, and reuses
    existing computations whenever possible. Thus, a Matrix
    object is "aware" of linear algebra theorems.

    Each Computable Property can be either:
     1. unset (default), in which case the corresponding
        function will be called to compute it.
     2. set to a callable, in which case that callable
        will be called to compute it.
     3. set to a value, which will be returned immediately.

    List of Computable Properties:
     - complex        : `True` if $M$ has complex entries
     - conjugate      : The complex conjugate of $M$
     - cond_2         : Condition number in 2-norm
     - determinant    : `det(M)`
     - diagonal       : `True` if $M$ is zero except for main diag.
     - diagonalizable : `True` if $M = P D P^{-1}$
     - eigenvalues    : The eigenvalues of $M$
     - eigenvectors   : The right eigenvectors of M
     - hermitian      : `True` if conjugate transpose of $M$ equals $M$
     - inverse        : if it exists, M^{-1}
     - invertible     : `True` if M^{-1} exists
     - lin_solver     : A function to solve Ax = b for x.
     - norm_2         : The 2-norm of the matrix
     - normal         : `True` if $M M^H = M^H M$
     - nullity        : `dim(Kernel(M))`
     - orthogonal     : `True` if $M^T = M^{-1}$
     - posdef         : `True` if $M$ is positive definite
     - qr             : The `(Q, R)` factorization of $M$
     - rank           : `dim(Col(M))`
     - sigmas         : Singular values of $M$
     - sparsity       : The fraction of zero entries in $M$
     - square         : `True` if $M$ is square
     - svd            : The SVD `(U, S, V)` of $M$
     - symmetric      : `True` if $M = M^T$
     - trace          : The trace of $M$
     - transpose      : $M^T$
     - triangular_L   : `True` if $M$ is lower triangular.
     - triangular_U   : `True` if $M$ is upper triangular.
     - unitary        : `True` if $M^H = M^{-1}$

    Non-Mathematical:
     - sparse         : `True` if the matrix is stored as a
                        SciPy Sparse object.
     - to_dense       : A dense representation of $M$
     - to_sparse      : A sparse representation of $M$
    """

    def set(self, prop, value):
        self._computed[prop] = value

    def __init__(self, arr):
        """Initialize a Matrix object.

        The argument `arr` can be a Numpy array, Scipy sparse matrix, list of
        Vector objects, or an object convertible to a Numpy array.

        If the resulting array is of dimension 1 in either axis, the
        Matrix automatically becomes a Vector.
        """
        if isinstance(arr, Matrix):
            arr = arr.entries
        if issparse(arr):
            # This is a SciPy sparse object. Preserve it.
            self.entries = arr
        elif (type(arr) is list or type(arr) is tuple) and len(arr) > 0 and \
                isinstance(arr[0], Vector):
            # This is a list of Vectors. Join them into a Matrix.
            any_complex = sum(v.is_complex() for v in arr) > 0
            if arr[0].nr == 1:
                # Treat all as row vectors
                self.entries = np.empty(
                    shape=(len(arr), len(arr[0])),
                    dtype=np.complex_ if any_complex else np.double)
                for j in range(len(arr)):
                    if len(arr[j]) != len(arr[0]):
                        raise MatrixSizeException(
                            arr[j], arr[0], "initialize Matrix from rows")
                    self.entries[j, :] = arr[j].entries.ravel()
            else:
                # Treat all as column vectors
                self.entries = np.empty(
                    shape=(len(arr[0]), len(arr)),
                    dtype=np.complex_ if any_complex else np.double)
                for j in range(len(arr)):
                    if len(arr[j]) != len(arr[0]):
                        raise MatrixSizeException(
                            arr[j], arr[0], "initialize Matrix from rows")
                    self.entries[:, j] = arr[j].entries.ravel()
        else:
            # This is another kind of object, hopefully numpy can handle it.
            self.entries = np.array(arr)
        if len(self.entries.shape) == 1:
            # Reshape ambiguous arrays into column vectors.
            self.entries = self.entries.reshape((len(self.entries), 1))
        if min(self.entries.shape) == 1 and not isinstance(self, Vector):
            # Automatically become a Vector if has a dimension of 1
            self.__dict__ = Vector(arr).__dict__
            self.__class__ = Vector

        # Initialize a blank dictionary of Computed Properties
        self._computed = {}

    """Directly accessible properties of the matrix"""

    @property
    def size(self) -> tuple:
        """The shape of the underlying matrix representation."""
        return self.entries.shape

    @property
    def nr(self) -> int:
        """The number of rows in the Matrix."""
        return self.size[0]

    @property
    def nc(self) -> int:
        """The number of columns in the Matrix."""
        return self.size[1]

    def __str__(self):
        return str(self.entries)

    def __repr__(self):
        return "{}x{} Matrix ({} properties)".format(
            self.nr, self.nc, len(self._computed))

    """Computable Properties corresponding to "checks".
    All correspond to matrix "types" and thus are queried
    using a method called `is_<type>()`."""

    @_computable_property("sparse")
    def is_sparse(self):
        """`True` if the underlying matrix representation is SciPy Sparse."""
        return issparse(self.entries)

    @_computable_property("complex")
    def is_complex(self):
        """`True` if the matrix has any non-zero complex entry."""
        return np.iscomplex(self.entries).any()

    @_computable_property("diagonal")
    def is_diagonal(self, tolerance=1e-10):
        """`True` if the non-diagonal elements of the matrix are within
        `tolerance` of zero."""
        check = False
        if self._computed.get("triangular_L", False) and \
                self._computed.get("triangular_U", False):
            return True
        if self.is_square():
            if self.is_sparse():
                check = (self.entries -
                         dia_array(
                             (self.entries.diagonal().T, 0),
                             shape=(self.nr, self.nc))).count_nonzero() == 0
            else:
                # Fast check for square matrices, @Daniel F on Stackoverflow
                test = self.entries.reshape(-1)[:-1].reshape(
                    self.nr-1, self.nc+1)
                check = ~np.any(test[:, 1:])
        else:
            check = np.allclose(
                self.entries - np.diag(self.entries, 0), 0, atol=tolerance)
        if check:
            # Automatically convert to a DiagonalMatrix
            self.entries = dia_array((self.diagonal(0).entries.T, 0),
                                     shape=(self.nr, self.nc))
            self.__class__ = DiagonalMatrix
        return check

    @_computable_property("diagonalizable")
    def is_diagonalizable(self, tolerance=1e-10):
        """`True` if the matrix can be diagonalized, $M = P D P^{-1}$.
        Does not necessarily compute the diagonalization. Densifies."""
        if not self.is_square():
            return False
        if self.is_sparse():
            return self.to_dense().is_diagonalizable()
        if "eigenvalues" in self._computed:
            if len(uniquetol(self.eigenvalues(), tolerance)) == self.nc:
                # N distinct eigenvalues is sufficient for diagonalizability.
                return True
        if self.is_normal():
            # Normal matrices are diagonalizable.
            return True
        # Check the linear independence of the eigenvectors through the SVD
        return Matrix(self.eigenvectors()).rank(tolerance) == self.nc

    @_computable_property("hermitian")
    def is_hermitian(self, tolerance=1e-10):
        """`True` if the matrix is Hermitian, $M^H = M$."""
        if not self.is_square():
            return False
        if self.is_complex():
            # Call the SciPy ishermitian function
            return ishermitian(self.entries, atol=tolerance)
        # For real matrices, equivalent to symmetric
        return self.is_symmetric()

    @_computable_property("invertible")
    def is_invertible(self, tolerance=1e-10):
        """`True` if the matrix has an inverse $M^{-1} M = Id$.
        Does not compute the inverse."""
        if "inverse" in self._computed:
            return True
        if "sigmas" in self._computed:
            # No zero singular values => invertible
            return np.isclose(self.sigmas(), 0, atol=tolerance).sum() == 0
        if (self.is_triangular_L() or self.is_triangular_U()) or \
                (not self.is_sparse() and "eigenvalues" in self._computed):
            # No zero eigenvalues => invertible
            return np.isclose(self.eigenvalues(), 0, atol=tolerance).sum() == 0
        # Square matrices with a finite condition number are invertible
        return self.is_square() and self.cond_2() < 1 / EPS

    @_computable_property("normal")
    def is_normal(self, tolerance=1e-10):
        """`True` if the matrix satisfies the normal equation $M^H M = M M^H$.
        Computes the conjugate transpose."""
        if not self.is_square():
            return False
        CT = self.conjugate().transpose()
        return (self @ CT).equals((CT @ self), tolerance)

    @_computable_property("orthogonal")
    def is_orthogonal(self, tolerance=1e-10):
        """`True` if the matrix is orthogonal, $M^T = M^{-1}$.
        Computes the transpose and identifies the inverse if `True`."""
        if not self.is_square():
            raise MatrixTypeException("orthogonality check", "square")
        prod = self @ self.transpose()
        check = prod.equals(Identity(prod.nc), tolerance)

        if check:
            # The inverse is now known.
            self.set("inverse", self.transpose())
            if not self.is_complex():
                self.set("unitary", True)
            self.transpose().set("orthogonal", True)
            self.transpose().set("inverse", self)
        return check

    @_computable_property("posdef")
    def is_posdef(self):
        """`True` if the matrix is positive-definite.
        Computes the eigenvalues. Densifies."""

        if self.is_sparse():
            return self.to_dense().is_posdef()
        return self.is_symmetric() and np.all(self.eigenvalues() > 0)

    def is_square(self):
        """`True` if the matrix is square."""
        return self.nr == self.nc

    @_computable_property("symmetric")
    def is_symmetric(self, tolerance=1e-10):
        """`True` if the matrix is symmetric, $M^T = M$.
        Computes the transpose only if the matrix is sparse."""
        if not self.is_square():
            return False
        else:
            if self.is_sparse():
                return (abs(self.entries
                            - self.transpose().entries) > tolerance).nnz == 0
            # Call SciPy issymmetric
            return issymmetric(self.entries, atol=tolerance)

    @_computable_property("triangular_L")
    def is_triangular_L(self, tolerance=1e-10):
        """`True` if the matrix is lower (left) triangular."""
        if self.is_sparse():
            return (abs(self.entries
                        - sparse_tril(self.entries)) > tolerance).sum() == 0
        return np.allclose(self.entries, np.tril(self.entries), atol=tolerance)

    @_computable_property("triangular_U")
    def is_triangular_U(self, tolerance=1e-10):
        """`True` if the matrix is upper (right) triangular."""
        if self.is_sparse():
            return (abs(self.entries
                        - sparse_triu(self.entries)) > tolerance).sum() == 0
        return np.allclose(self.entries, np.triu(self.entries), atol=tolerance)

    @_computable_property("unitary")
    def is_unitary(self, tolerance=1e-10):
        """`True` if the matrix is unitary, $M^H = M^{-1}$.
        Computes the conjugate transpose and identifies the inverse if True."""
        if not self.is_square():
            raise MatrixTypeException("unitary check", "square")
        if not self.is_complex():
            return self.is_orthogonal()
        CT = self.conjugate().transpose()
        prod = self @ CT
        check = prod.equals(Identity(prod.nc), tolerance)

        if check:
            # The inverse is now known.
            self.set("inverse", CT)
            CT.set("unitary", True)
            CT.set("inverse", self)
        return check

    """Computable Properties of the matrix that are not checks."""

    @_computable_property("cond_2")
    def cond_2(self):
        """Compute the condition number $\\kappa_2 for inversion of the matrix.
        Computes the singular values."""
        sigmas = self.sigmas()
        return sigmas[0] / sigmas[-1]

    @_computable_property("determinant")
    def determinant(self):
        """Compute the determinant of the matrix. Densifies.
        Computes the eigenvalues of triangular matrices."""
        if not self.is_square():
            raise MatrixTypeException("determinant", "square")
        if self.is_sparse():
            # For now, we have to go to a dense representation
            return self.to_dense().determinant()
        else:
            if "eigenvalues" in self._computed or \
                    (self._computed.get("triangular_L", False) or
                     self._computed.get("triangular_U", False)):
                # Use the eigenvalues if available
                return np.prod(self.eigenvalues())
            return np.linalg.det(self.entries)

    @_computable_property("eigenvalues")
    def eigenvalues(self, tolerance=1e-10):
        """Compute the eigenvalues of the matrix.
        Does not compute the eigenvectors, unless the matrix is sparse.
        Checks whether $M$ is Hermitian, unless the matrix is triangular."""
        if not self.is_square():
            raise MatrixTypeException("eigenvalues", "square")
        if self.is_triangular_L(tolerance) or self.is_triangular_U(tolerance):
            # Triangular matrix eigenvalues are on the diagonal.
            return np.sort(self.diagonal(0))
        if self.is_sparse():
            # Use sparse algorithms
            self.eigenvectors(tolerance)
            return self.eigenvalues()
        if self.is_hermitian():
            # Use the specialized algorithm for Hermitian matrices.
            return eigvalsh(self.entries, check_finite=False)
        vals = eigvals(self.entries, check_finite=False)
        if np.iscomplex(vals).any():
            return np.sort_complex(vals)
        return np.sort(vals)

    @_computable_property("eigenvectors")
    def eigenvectors(self, tolerance=1e-10, sparse_k="max"):
        """Compute the eigenvectors of the matrix.
        Computes the eigenvalues. Checks whether $M$ is Hermitian.

        Note that if the matrix has a sparse representation, then
        only `sparse_k` many eigenvalues/vectors can be computed!
        The `max` is 90% of the dimension of the matrix.
        """
        if not self.is_square():
            raise MatrixTypeException("eigenvectors", "square")
        if self.is_sparse():
            sparse_k = min(self.nr - 1, int(0.9 * self.nr))
            # Use the sparse algorithms
            if self.is_hermitian():
                # Use the specialized algorithm for Hermitian matrices
                vals, vecs = sparse_eigsh(self.entries, k=sparse_k)
            else:
                vals, vecs = sparse_eigs(self.entries, k=sparse_k)
            if not np.iscomplex(vals).any():
                order = np.argsort(vals)
                # Report the eigenvalues and vectors in increasing order
                vals = vals[order]
                vecs = vecs[:, order]
        else:
            if self.is_hermitian():
                # Use the specialized algorithm for Hermitian matrices.
                vals, vecs = eigh(self.entries, check_finite=False)
            else:
                vals, vecs = eig(self.entries, check_finite=False)
                if not np.iscomplex(vals).any():
                    order = np.argsort(vals)
                    # Report the eigenvalues and vectors in increasing order
                    vals = vals[order]
                    vecs = vecs[:, order]
        self.set("eigenvalues", vals)
        return [Vector(vecs[:, j]) for j in range(len(vals))]

    @_computable_property("norm_2")
    def norm_2(self):
        """Compute the 2-norm of the matrix $\\lVert M \\rVert_2."""
        if self._computed.get("orthogonal", False) or \
                self._computed.get("unitary", False):
            # Unitary and Orthogonal matrices have norm 1.
            return 1.0
        if "sigmas" in self._computed:
            # The norm is the largest singular value.
            return self.sigmas()[0]
        if self.is_sparse():
            return sparse_norm(self.entries, ord=2)
        return np.linalg.norm(self.entries, ord=2)

    @_computable_property("nullity")
    def nullity(self, tolerance=1e-10):
        """Compute the nullity as the number of columns minus the rank."""
        return self.nc - self.rank()

    @_computable_property("rank")
    def rank(self, tolerance=1e-10):
        """Compute the rank of the matrix.
        Computes the SVD if no inverse or eigenvalues are known."""
        if "invertible" in self._computed or "inverse" in self._computed:
            # Invertible => full rank
            return self.nc
        if not self.is_sparse() and "eigenvalues" in self._computed:
            # Number of nonzero eigenvalues
            return np.logical_not(
                np.isclose(self.eigenvalues(), 0, atol=tolerance)).sum()
        # Number of nonzero signular values
        _, S, _ = factor.svd(self)
        return np.logical_not(
            np.isclose(S.diagonal(0).entries, 0, atol=tolerance)).sum()

    @_computable_property("sigmas")
    def sigmas(self):
        """Compute the singular values of the matrix.
        Computes the SVD if not positive-definite."""
        if "posdef" in self._computed and self.is_posdef():
            return np.flip(np.sqrt(self.eigenvalues()))
        factor.svd(self)
        return self.sigmas()

    @_computable_property("sparsity")
    def sparsity(self, tolerance=1e-10):
        """Compute the sparsity of the matrix as the fraction of zero
        entries out of all entries in the matrix."""
        if self.is_sparse():
            # Use the sparse methods if available
            return len(self.entries.nnz) / (self.nr * self.nc)
        return np.isclose(
            self.entries, 0, atol=tolerance).sum() / (self.nr * self.nc)

    @_computable_property("trace")
    def trace(self):
        """Compute the trace of the matrix."""
        if self.is_sparse():
            return self.entries.trace()
        if "eigenvalues" in self._computed:
            return np.sum(self.eigenvalues())
        return np.trace(self.entries)

    """Getter functions for elements of the matrix."""

    def row(self, number):
        """Return the 1-indexed row."""
        if number < 0:
            return Vector(self.entries[number, :])
        return Vector(self.entries[number-1, :])

    def column(self, number):
        """Return the 1-indexed column."""
        if number < 0:
            return Vector(self.entries[:, number])
        return Vector(self.entries[:, number-1])

    def diagonal(self, number):
        """Return the k-th diagonal."""
        if self.is_sparse():
            return Vector(self.entries.diagonal(number))
        return Vector(np.diagonal(self.entries, number))

    def entry(self, row, col):
        """Return the 1-indexed entry $M_{ij}$."""
        return self.entries[row if row < 0 else row - 1,
                            col if col < 0 else col-1]

    def equals(self, M, tolerance=1e-10):
        """`True` if the matrix equals another elementwise,
        within a tolerance."""
        if type(M) is int or type(M) is float:
            return np.allclose(self.entries, M, atol=tolerance)
        if self.is_sparse() or M.is_sparse():
            # Use subtraction if one or both are sparse
            return np.allclose(self.entries - M.entries, 0, atol=tolerance)
        return np.allclose(self.entries, M.entries, atol=tolerance)

    def __eq__(self, M):
        return self.equals(M)

    """Transformations of the matrix."""

    @_computable_property("dense_mat")
    def to_dense(self):
        """Switch from a sparse to a dense repsresentation."""
        if self.is_sparse():
            DM = Matrix(self.entries.todense())
            DM._computed = self._computed
            # Remove properties that are best recomputed
            for k in ("cond_2", "dense", "eigenvalues", "eigenvectors", "rank",
                      "invertible", "inverse", "sigmas", "sparse", "svd",
                      "transpose"):
                DM._computed.pop(k, None)
            DM.set("sparse_mat", self)
            return DM
        return self

    @_computable_property("sparse_mat")
    def to_sparse(self, tolerance=1e-10):
        """Switch from a dense to a sparse representation."""
        if self.is_sparse():
            return self
        if self.is_diagonal():
            return DiagonalMatrix(self.diagonal(0))
        return Matrix(csc_array(self.entries))

    @_computable_property("conjugate")
    def conjugate(self):
        """Compute the complex conjugate of the matrix."""
        if self.is_sparse():
            Conj = Matrix(self.entries.conjugate())
        else:
            Conj = Matrix(np.conjugate(self.entries))
        Conj.set("conjugate", self)
        if "triangular_L" in self._computed:
            Conj.set("triangular_L", self.is_triangular_L())
        if "triangular_U" in self._computed:
            Conj.set("triangular_U", self.is_triangular_U())
        if "diagonal" in self._computed:
            Conj.set("diagonal", self.is_diagonal())
        return Conj

    @_computable_property("inverse")
    def inverse(self):
        """Compute the inverse of the matrix. Checks invertibility."""
        if not self.is_invertible():
            raise MatrixTypeException("inverse", "invertible")
        if "qr" in self._computed:
            # Use the QR if it exists
            Q, R = self.qr()
            Inv = R.inverse() @ Q.transpose()
        elif "svd" in self._computed:
            # Use the SVD if it exists
            U, S, V = self.svd()
            Inv = V.conjugate().transpose() @ S.inverse() \
                @ U.conjugate().transpose()
        else:
            # Nothing known, compute inverse
            Inv = Matrix(inv(self.entries))
        Inv.set("inverse", self)
        # Set known properties
        if "determinant" in self._computed:
            Inv.set("determinant", 1.0 / self.determinant())
        if "eigenvalues" in self._computed:
            Inv.set("eigenvalues", np.flip(1.0 / self.eigenvalues()))
        # Set the lin solver to the found inverse
        self.set("lin_solver", lambda v: solve.invert(self, v))
        return Inv

    @_computable_property("lin_solver")
    def lin_solver(self):
        """Return a method which solves $Ax=b$ for $x$.
        See `solve.automatic`."""
        return solve.automatic(self, None, get_method=True)

    @_computable_property("qr")
    def qr(self):
        """Compute the QR factorization of the matrix. See `factor.qr`."""
        return factor.qr(self)

    @_computable_property("svd")
    def svd(self):
        """Compute the SVD factorization of the matrix. See `factor.svd`."""
        return factor.svd(self)

    @_computable_property("transpose")
    def transpose(self):
        """Compute the transpose of the matrix."""
        if self.is_sparse():
            T = Matrix(self.entries.transpose())
        else:
            T = Matrix(self.entries.T)
        # Copy properties that are identical for the transpose
        T._computed = {prop: self._computed[prop] for prop in [
            "complex", "cond_2", "determinant", "diagonal"
            "eigenvalues", "eigenvectors", "invertible", "rank",
            "sigmas", "symmetric"
        ] if prop in self._computed}
        T.set("transpose", self)
        if "triangular_L" in self._computed:
            T.set("triangular_U", self.is_triangular_L())
        if "triangular_U" in self._computed:
            T.set("triangular_L", self.is_triangular_U())
        if "svd" in self._computed:
            U, S, V = self.svd()
            T.set("svd", (V.transpose(), S, U.transpose()))
        return T

    """Operation implementations."""

    def __mul__(self, M):
        """Multiplication by a constant."""
        if type(M) is int or type(M) is float:
            return Matrix(M * self.entries)
        raise TypeError("Use of * to multiply by a {}.".format(type(M)))

    def __rmul__(self, M):
        return self.__mul__(M)

    def __neg__(self):
        return self.__mul__(-1)

    def __matmul__(self, M):
        """Matrix multiplication."""
        if not isinstance(M, Matrix):
            # Try to promote to Matrix object
            M = Matrix(M)
        if self.nc == M.nr:
            return Matrix(self.entries @ M.entries)
        raise MatrixSizeException(self, M, "multiply")

    def __add__(self, M):
        """Addition"""
        if self.nc != M.nc or self.nr != M.nr:
            raise MatrixSizeException(self, M, "addition")
        if not isinstance(M, Matrix):
            # Try to promote to Matrix object
            M = Matrix(M)
        return Matrix(self.entries + M.entries)

    def __radd__(self, M):
        return self.__add__(M)

    def __sub__(self, M):
        """Subtraction"""
        if self.nc != M.nc or self.nr != M.nr:
            raise MatrixSizeException(self, M, "subtraction")
        if not isinstance(M, Matrix):
            # Try to promote to Matrix object
            M = Matrix(M)
        return Matrix(self.entries - M.entries)

    def __rsub__(self, M):
        return (-self).__add__(M)


class DiagonalMatrix(Matrix):
    """A DiagonalMatrix is a Matrix whose only nonzero entries
    are on the primary diagonal."""

    def __init__(self, diagonal):
        """Initialize a DiagonalMatrix from an array of the diagonal."""
        if isinstance(diagonal, Vector):
            diagonal = diagonal.entries
        # Use the SciPy sparse diagonal implementation.
        super().__init__(dia_array((diagonal, 0),
                                   shape=(len(diagonal), len(diagonal))))
        self.set("diagonal", True)

    def __getitem__(self, number):
        """Return the 1-indexed entry along the diagonal."""
        if number < 0:
            return self.entries.diagonal()[number]
        return self.entries.diagonal()[number-1]

    @_computable_property("invertible")
    def is_invertible(self, tolerance=1e-10):
        """`True` if there are no zeros on the diagonal."""
        if "inverse" in self._computed:
            return True
        return np.isclose(
            self.entries.diagonal(), 0, atol=tolerance).sum() == 0

    @_computable_property("determinant")
    def determinant(self):
        """Compute the determinant as the product of the diagonal."""
        return np.prod(self.entries.diagonal())

    @_computable_property("eigenvalues")
    def eigenvalues(self, tolerance=1e-10):
        """Compute the eigenvalues, equivalent to sorting the diagonal."""
        return np.sort(self.entries.diagonal())

    @_computable_property("inverse")
    def inverse(self):
        """Compute the inverse as the reciprocal of the diagonal."""
        if self.is_invertible():
            return DiagonalMatrix(1.0 / self.entries.diagonal())
        raise MatrixTypeException("inverse", "invertible")

    @_computable_property("trace")
    def trace(self):
        """Compute the trace of the matrix."""
        return np.sum(self.diagonal().entries)


class Identity(DiagonalMatrix):
    """The Identity matrix is a DiagonalMatrix of ones."""

    def __init__(self, n):
        """Initialize the Identity of size n."""
        Matrix.__init__(self, identity(n))


class Vector(Matrix):
    """A Vector is a Matrix of size 1 in one dimension."""

    def __init__(self, arr, row=False):
        """Initialize a Vector object from a Numpy array.
        If the array has only one dimension, defaults to being a column vector.
        A row vector can be forced by setting row=True."""
        if isinstance(arr, Matrix):
            arr = arr.entries
        arr = np.array(arr)
        if len(arr.shape) == 1:
            # One-dimensional array.
            if row:
                super().__init__(arr.reshape(1, len(arr)))
            else:
                super().__init__(arr.reshape(len(arr), 1))
        else:
            # 2-dimensional array, initialize the Matrix.
            super().__init__(arr)

        if min(self.nr, self.nc) != 1:
            raise ValueError(
                "A Vector cannot have size {}x{}".format(self.nr, self.nc))

    def __len__(self):
        """Return the number of entries in the Vector."""
        return max(self.nr, self.nc)

    def __getitem__(self, number):
        """Returns the 1-indexed entry in the Vector."""
        if number < 0:
            return self.entries[number]
        return self.entries[number-1]

    def __repr__(self):
        return "{}x{} Vector ({} properties)".format(
            self.nr, self.nc, len(self._computed))

    @_computable_property("unit")
    def unit(self):
        """Compute the unit vector in the direction of this Vector.
        Computes the 2-norm."""
        return Vector(self.entries / self.norm_2())
