"""MathMat Random Matrix Toolbox"""

from .mathmat import Matrix
import numpy as np
from scipy.fft import dct


class Gaussian(Matrix):
    """A Gaussian matrix has entries drawn from a standard normal."""

    def __init__(self, nr, nc):
        """Initialize a new Gaussian matrix with the given size."""
        super().__init__(np.random.standard_normal(size=(nr, nc)))


class FFTSketchedMatrix(Matrix):
    """A sketching matrix based on the FFT / DCT."""

    def __init__(self, M: Matrix):
        """Sketch an existing Matrix."""
        signs = np.sign(np.random.standard_normal(size=(1, M.nc)))
        arr = signs * M.entries
        if M.is_complex():
            arr = np.fft.fft2(arr)
        else:
            arr = dct(arr, type=2)
        super().__init__(arr)


def approx_hmt(M: Matrix, r: int):
    """Compute a rank `r` approximation using the HMT algorithm."""
    S = Gaussian(M.nr, r)
    MS = M @ S
    Q, _ = MS.qr()
    return Q @ Q.transpose() @ M
