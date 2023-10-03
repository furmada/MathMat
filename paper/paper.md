---
authors:
- name: Adam Furman
  orcid: 0000-0002-6675-1064
  affiliation: 1
affiliations:
- name: University of Oxford Mathematical Institute, Oxford, United Kingdom
  index: 1
bibliography:
- pysci.bib
date: 29 September 2023
title: "MathMat: A Mathematically Aware Matrix Library"
tags:
  - Python
  - Linear Algebra
  - Matrix
---

# Summary

The MathMat library provides a mathematically aware
toolbox for working with matrices. A matrix is represented in two parts: the
entries of the matrix, and a collection of known properties of the
matrix. Checking or computing a property stores it in the matrix object,
allowing it to be reused without additional computational cost. The methods for computing properties are stateful, in that they respond to what other properties have already been computed. For example, if a matrix is triangular, the eigenvalues are the entries on the diagonal. Thus, when computing eigenvalues of a matrix known to be triangular, MathMat returns the diagonal instead of
performing an additional expensive computation.

# Statement of Need

The commonly used libraries NumPy [@NumPy] and SciPy [@SciPy] both have ways to represent matrices numerically, and methods of computing or checking various matrix properties. However, the provided representations of matrices have no sense of their own \"matrix-ness,\" that is, they are collections of data with no awareness of any mathematical properties associated with that data.

MathMat is useful as a wrapper for the matrix-related functionality of NumPy and SciPy. In addition to the aforementioned leveraging of mathematical properties to simplify computations, MathMat also unifies working with dense and sparse matrices, allowing for automatic efficient data storage. MathMat is written in a way designed to be extensible and customizable for a variety of potential research contexts, from numerical analysis to data processing.

In addition to mathematical awareness of matrix properties,
MathMat contains implementations of some matrix
algorithms, specifically Krylov subspace methods. The package has
visualization capabilities as well, focusing on generating plots that
are aesthetically pleasing and clearly visible within academic papers,
with focus again towards those useful in numerical linear algebra. Thus, MathMat is also a convenience library for any project involving matrices.

# Computable Properties

Computable Properties of a `Matrix` object fall into two categories: *checks* and *values*. A \"check\" is a property which is either `True` or `False`, for example, whether the matrix is symmetric or not. A \"value\" is a Property which returns a non-boolean type, for example, the eigenvalues obtained by calling `eigenvalues()`. The key feature of Computable Properties is that, once computed, they are never re-computed for the same `Matrix` instance.

## Efficiency through Computable Properties

The philosophy of MathMat is to increase efficiency for
multiple computations on the same matrix. Efficiency of computations is
preferred over memory consumption: storing properties will never be more
efficient in terms of storage requirements. Under this philosophy, it is
acceptable for an individual initial computation to be more
computationally complex, as long as that increased complexity produces
knowledge about the matrix that can be reused later.

An illustrative example of the MathMat philosophy is the
`eigenvalues` method, which does many checks \"hoping\" that it will discover that the `Matrix` has properties which simplify the computation of eigenvalues.
After checking that the matrix satisfies the requirement of being
square, it is checked whether the matrix is triangular. Knowing a matrix
is triangular is a strong statement, making it much easier to find
eigenvalues, but also allowing the use of faster algorithms for solving
linear systems. The check is \"worthwhile\" in that it increases what is
known about the matrix for future tasks.

A call to MathMat's `eigenvalues` may take substantially
longer than directly invoking `scipy.linalg.eigvals`, but it is also
much more informative. For example, if the matrix was found to be upper
triangular then computing the determinant, checking for invertability,
and solving the linear equation $Mx = b$ will all be automatically
optimized, without the check for triangularity being performed again.

Another consistent advantage of MathMat is the equivalence
of sparse and dense matrix representations. While NumPy arrays and SciPy
sparse objects are often compatible, many NumPy specific functions
cannot handle a mix of the two. MathMat abstracts the
distinction away, automatically adapting when a sparse representation is
used, and employing the appropriate sparse algorithms to take full
advantage of the sparse format. An example is the `is_diagonal` method
(Appendix \ref{code:is_diagonal}), which uses a check based on the
`count_nonzero()` method of sparse objects, or one of two different
checks for dense matrices. If the matrix is found to be diagonal, it is automatically stored in a more data-efficient format.

# Factorizations, Solvers, and Visualizations

MathMat implements several common types of matrix factorizations.
The `Matrix` instances of factored matrices have Computable Properties `set` immediately on creation, meaning that the mathematical knowledge of the factorization is preserved and can be automatically used to simplify potential further computations. The existence of a QR factorization, for example, is detected when solving a least-squares problem and a more efficient triangular solver is used. 

MathMat likewise implements several methods for solving $Ax = b$. In keeping with the philosophy of the module, the `automatic` solver method uses the properties of the matrix $A$ to pick the most efficient algorithm to solve the system. It prefers using an inverse if one exists, under the assumption that a matrix with a set `inverse` is one for which $M^{-1}$ can be computed in a numerically stable way, and will then check triangularity and the existence of a QR or LU decomposition before falling back on using GMRES [@Saad1986]. MathMat also implements the Sketched GMRES algorithm of Nakatsukasa and Tropp, which is subject to active research [@Nakatsukasa2022].

Finally, the visualization submodule provides aesthetically pleasing plots that fit well within mathematical publications with minimum adjustment required. MatPlotLib is used as the backend for producing the plots, and several aesthetic adjustments are made to the default configuration of
`matplotlib`.

# References