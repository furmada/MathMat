"""MathMat is a matrix library built on NumPy that tries to reuse
existing data from past computations whenever possible.

The functionality is built around the Matrix class, which represents
a matrix as an array of data plus a dictionary of properties. As
properties are checked and/or computed, existing information is reused
when possible to speed up new computations.

MathMat also provides utilities for performing matrix factorizations
via the `factor` submodule, solving linear systems via the `solve`
submodule, and visualizations using the `plot` submodule.
"""

from .mathmat import EPS
from .mathmat import uniquetol
from .mathmat import Vector
from .mathmat import Matrix, DiagonalMatrix, Identity
from .mathmat import MatrixSizeException, MatrixTypeException


__author__ = "1071298"
__credits__ = ["1071298"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "1071298"
__email__ = "proof_inquiries@fermat.fr"
__status__ = "Development"
