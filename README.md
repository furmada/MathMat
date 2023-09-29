# MathMat
A mathematically aware matrix library.

### What is MathMat?

The `mathmat` library specializes NumPy and SciPy to provide a mathematically aware toolbox for working with numerical linear algebra problems. The core idea is to represent a matrix in two parts: the data, comprising the entries of the matrix, and a collection of known properties about the matrix. Checking or computing a property stores it in the matrix object, allowing it to be reused without additional computational cost. In addition, the methods for computing properties are stateful, in that they respond to what other properties have already been computed.  For example, if a matrix is triangular, the eigenvalues are the entries on the diagonal. Thus, when computing eigenvalues of a matrix known to be triangular, `mathmat` returns the diagonal instead of performing an additional expensive computation.

In addition to mathematical awareness of matrix properties, `mathmat` contains implementations of some matrix algorithms, specifically Krylov subspace methods. The package has visualization capabilities as well, focusing on generating plots that are aesthetically pleasing and clearly visible within academic papers, with focus again towards those useful in numerical linear algebra.

`mathmat` is organized into one library with five submodules, including the main module. Each submodule covers a specific range of functionality, outlined here.

- `mathmat`: The core definitions, especially the `Matrix` class which defines the mathematically-aware matrix object.
- `mathmat.factor`: Methods for computing matrix factorisations, such as the SVD and QR decomposition.
- `mathmat.random`: Random matrix definitions and algorithms.
- `mathmat.solve`: Methods for solving the linear system $A x = b$ for $x$.
- `mathmat.vis`: Visualization tools for matrices and other data.

### How does MathMat work?

At the core of `mathmat` is the Matrix class. Each `Matrix` instance is meant to be immutable, thus the data is not meant to be change after it is initialized. The Matrix can be initialized using a NumPy array, SciPy sparse matrix, list of `Vector` objects, or any object which can be parsed successfully by `np.array`. Regardless of what type of input is given to the constructor, the `Matrix` object will have an `entries` attribute which is either a NumPy array or a SciPy sparse matrix after initialization.

**Computable Properties**

Computable Properties fall into two categories: *checks* and *values*. A "check" is a property which is either `True` or `False`, for example, whether the matrix is symmetric or not. All checks correspond to methods of the *Matrix* object starting with the prefix *is_* (as in *is_symmetric()*). Note that all Computable Properties are callable methods (as opposed to true properties of the object, or methods using the `@property` decorator) to highlight that computing them is an active process which may be time and resource intensive. A "value" is a Property which returns a non-boolean type, for example, the eigenvalues obtained by calling `eigenvalues()`.

An alphabetized table of the Computable Properties of a `Matrix` $M$ is given below. All Computable Properties of type `C` are *checks* and their methods have the prefix `is_*`, those of type `V` are *values*.

| Property Name       | Type | Description                                                                   |
|---------------------|------|-------------------------------------------------------------------------------|
| `complex`           | `C`  | `True` if $M$ has nonzero complex entries.                                   |
| `conjugate`         | `V`  | The complex conjugate $M^*$ of $M$.                                          |
| `cond_2`            | `V`  | The condition number $\kappa_2(M)$.                                          |
| `determinant`       | `V`  | The determinant $| M |$.                                                   |
| `diagonal`          | `C`  | `True` if $M$ only has non-zero entries on the main diagonal.                |
| `diagonalizable`    | `C`  | `True` if $M$ is similar to a diagonal matrix $D$ such that $M = P D P^{-1}$. |
| `eigenvalues`       | `V`  | The eigenvalues $\lambda$ satisfying $Mv = \lambda v$ sorted by increasing magnitude. |
| `eigenvectors`      | `V`  | The eigenvectors $v$ satisfying $Mv = \lambda v$ sorted by the magnitude of their corresponding eigenvalues. |
| `hermitian`         | `C`  | `True` if $M$ is Hermitian and thus $M^H = M^{-1}$.                          |
| `inverse`           | `V`  | The inverse $M^{-1}$, raises an error if $M$ is not invertible.               |
| `invertible`        | `C`  | `True` if $M$ is invertible.                                                 |
| `lin_solver`        | `V`  | A callable which, given a `Vector` $b$, solves $Mx = b$ for $x$.             |
| `norm_2`            | `V`  | The 2-norm of the matrix $\lVert M \rVert_2$.                                |
| `normal`            | `C`  | `True` if $M$ is normal and thus $M M^H = M^H M$.                             |
| `nullity`           | `V`  | The dimension of the kernel of $M$.                                          |
| `orthogonal`        | `C`  | `True` if $M$ is orthogonal and thus $M^\top = M^{-1}$.                      |
| `posdef`            | `C`  | `True` if $M$ is positive-definite, $x^\top M x > 0 \: \forall x$.           |
| `qr`                | `V`  | A `tuple` with the `(Q, R)` decomposition $M = QR$.                          |
| `rank`              | `V`  | The dimension of the column space of $M$.                                    |
| `sigmas`            | `V`  | The singular values of $M$ in decreasing magnitude.                           |
| `sparse`            | `C`  | `True` if the `entries` property of $M$ is a SciPy sparse object.             |
| `sparsity`          | `V`  | The fraction of entries of $M$ which are non-zero.                           |
| `square`            | `C`  | `True` if $M$ is square.                                                     |
| `svd`               | `V`  | A `tuple` with the `(U, S, V)` matrices of the singular value decomposition $M = U \Sigma V^H$. |
| `symmetric`         | `C`  | `True` if $M$ is symmetric, $M^\top  = M$.                                    |
| `to_dense`          | `V`  | A representation of $M$ using a NumPy array instead of a SciPy sparse object. |
| `to_sparse`         | `V`  | A representation of $M$ using a SciPy `csc_array` instead of a NumPy array.   |
| `trace`             | `V`  | The sum of the diagonal of $M$.                                              |
| `transpose`         | `C`  | The transpose $M^\top$ of $M$.                                               |
| `triangular_L`      | `C`  | `True` if $M$ is lower (left) triangular.                                     |
| `triangular_U`      | `C`  | `True` if $M$ is upper (right) triangular.                                    |
| `unitary`           | `C`  | `True` if $M$ is unitary, $M^H = M^{-1}$.                                    |

Some `mathmat` functions may store additional properties using the `set` method of the `Matrix`, but these are not accessible using methods.

**Factorisations**

The following matrix factorisations are implemented in the `mathmat.factor` package:

- `cholesky`: Cholesky factorisation of Hermitian matrices into $M = L L^H$ where $L$ is lower triangular.
- `diagonalization`: Diagonalization of diagonalizable matrices into $M = P D P^{-1}$.
- `lu`: Pivoted LU decomposition of square matrices into $M = P L U$ where $L$ is lower and $U$ is upper triangular.
- `qr`: QR factorisation $M = Q R$ where $R$ is upper triangular and $Q$ is orthogonal.
- `svd`: The singular value decomposition into $M = U \Sigma V^H$ where $U$ is unitary, $\Sigma$ is diagonal, and $V^H$ is orthogonal.

**Linear System Solvers**

The `mathmat.solve` submodule has several available methods for solving the linear system $Ax = b$. In keeping with the philosophy of the module, the cornerstone is the `automatic` method, which uses the properties of the matrix $A$ to pick the most efficient algorithm to solve the system. The implemented solvers are:

- `gmres`: The Generalized Minimum Residual algorithm of Saad and Schulz. This is a custom implementation, described in more detail in a later section.
- `sgmres`: The Sketched GMRES algorithm of Nakatsukasa and Tropp.
- `invert`: Solves the linear system $Ax = b$ by inverting $A$ explicitly.
- `lstsq`: Solves the least-squares, minimum norm problem $x = \min \lVert Ax - b \rVert_2$
- `lu`: Solves the linear system $Ax = b$ by computing the LU factorisation and then solving two triangular systems.
- `triangular`: Solves the system $Lx = b$ or $Ux = b$ for a triangular $L$ or $U$.

**Visualizations**

The goal of this submodule is to provide aesthetically pleasing plots that fit well within mathematical publications with minimum adjustment required. `matplotlib` is used as the backend for producing the plots.

Several adjustments are made to the default configuration of `matplotlib`. The library is configured to produce a new "pop-up" interactive window for each new figure, making it easier to see and export the output in a standard Python shell. The font for all plot text and math is changed to Computer Modern, the default `latex` typeface. Finally, the thickness of lines is slightly increased to yield better visibility when embedded into papers.

Each plot is an instance of the `Plot` class. A generic `Plot` object displays no data, but is capable of setting the title, $x$ and $y$ axis labels, and legend entries. A second generic class, `DataPlot`, extends `Plot` by further adding the ability to set logarithmic scales on either or both axes. All plots also accept any number of keyword arguments which are passed on to `matplotlib`. The following subclasses of `Plot` are intended to be used directly:

- `LinePlot`: Graphs array-like data as connected lines.
- `ScatterPlot`: Displays array-like data as disconnected points.
- `Histogram`: Shows a one-dimensional array as a histogram over a specified number of columns.
- `MagnitudePlot`: Visualizes the magnitude of the entries of a matrix $M$ on a greyscale plot, where darker shades correspond to higher magnitude.
- `SparsityPlot`: Like `MagnitudePlot`, but colors all non-zero entries (within a tolerance) black.
- `OrthogonalityPlot`: Visualizes the orthogonality of the columns of a matrix $M$ by computing $G = M^H M$ and then showing the `MagnitudePlot` of $G$.

Instantiating a `Plot` object does not immediately display it. Instead, one of two methods is used to generate the actual figure. If only one `Plot` needs to be displayed, `matlib.vis.single` is called, passing the `Plot` instance. If a grid of plots is desired, `matlib.vis.tiled` is called, passing a list of `Plot`s matching the shape of the desired grid, as well as an optional super-title.

### More Documentation

The `docs` folder contains complete documentation for the package.