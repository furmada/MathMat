"""MathMat Visualization Toolbox."""

import matplotlib.pyplot as plt
from matplotlib import use as use_mpl_backend
import numpy as np

from mathmat import Matrix, EPS

# Configure Matplotlib
use_mpl_backend("TkAgg")
params = {"ytick.color": "black",
          "xtick.color": "black",
          "axes.labelcolor": "black",
          "axes.edgecolor": "black",
          "axes.formatter.use_mathtext": True,
          "text.usetex": False,
          "font.family": "serif",
          "font.serif": "cmr10",
          "mathtext.fontset": "cm",
          "figure.dpi": 100,
          "figure.autolayout": True,
          "savefig.dpi": 300,
          "savefig.transparent": True,
          "savefig.bbox": "tight",
          "font.size": 12,
          "lines.linewidth": 2}
plt.rcParams.update(params)


class Plot:
    """A Plot is the base class for all visualizations.
    Do not use Plot directly, as it does not show any data.
    Instead, extend it or use a specific type of Plot."""

    def __init__(self, legend_labels=None, title=None, x_label=None,
                 y_label=None, **mpl_kwargs):
        if type(legend_labels) is not list \
                and legend_labels is not None:
            legend_labels = [legend_labels]
        self.legend_labels = legend_labels
        self.title = title
        self.x_label = x_label
        self.y_label = y_label
        self._mpl_kwargs = mpl_kwargs

    def plot_on(self, axes):
        """Show the Plot on the specified matplotlib Axes."""
        if self.legend_labels is not None:
            axes.legend(self.legend_labels)
        if self.title is not None:
            axes.set_title(self.title)
        if self.x_label is not None:
            axes.set_xlabel(self.x_label)
        if self.y_label is not None:
            axes.set_ylabel(self.y_label)
        axes.margins(x=0, y=0)


class DataPlot(Plot):
    """A DataPlot visualizes data. Do not use it directly, instead
    call LinePlot or ScatterPlot."""

    def __init__(self, Xs, Ys, formats=None, legend_labels=None, log_x=False,
                 log_y=False, title=None, x_label=None,
                 y_label=None, **mpl_kwargs):
        super().__init__(legend_labels, title, x_label, y_label, **mpl_kwargs)
        self.Xs = np.array(Xs)
        self.Ys = np.array(Ys)
        self.formats = formats
        self.log_x = log_x
        self.log_y = log_y
        self._plt_args = (self.Xs.T, self.Ys.T) if self.formats is None else \
            (self.Xs, self.Ys, self.formats)

    def plot_on(self, axes):
        if self.log_x:
            axes.set_xscale("log")
        else:
            axes.set_xscale("linear")
        if self.log_y:
            axes.set_yscale("log")
        else:
            axes.set_yscale("linear")
        super().plot_on(axes)


class LinePlot(DataPlot):
    """A LinePlot visualizes data as lines.
    `Xs` and `Ys` must be are array-like objects of equal length.
    Optionally, `formats` is an array of equal length containing
    format specifications for `matplotlib`.
    `log_x` and `log_y` are used to specify logarithmic scaling of axes."""

    def plot_on(self, axes):
        axes.plot(*self._plt_args, **self._mpl_kwargs)
        super().plot_on(axes)


class ScatterPlot(DataPlot):
    """A LinePlot visualizes data as points.
    `Xs` and `Ys` must be are array-like objects of equal length.
    Optionally, `formats` is an array of equal length containing
    format specifications for `matplotlib`.
    `log_x` and `log_y` are used to specify logarithmic scaling of axes."""

    def plot_on(self, axes):
        axes.scatter(*self._plt_args, **self._mpl_kwargs)
        super().plot_on(axes)


class Histogram(Plot):
    """A Histogram visualizes data by grouping it based on values.
    `log_x` and `log_y` are used to specify logarithmic scaling of axes."""

    def __init__(self, Xs, bins=None,
                 legend_labels=None, title=None, x_label=None, y_label=None,
                 **mpl_kwargs):
        super().__init__(legend_labels, title, x_label, y_label,
                         **mpl_kwargs)
        self.Xs = np.array(Xs)
        self.bins = min(len(Xs), 10) if bins is None else bins

    def plot_on(self, axes):
        axes.hist(self.Xs, self.bins, **self._mpl_kwargs)
        super().plot_on(axes)


class MagnitudePlot(Plot):
    """A MagnitudePlot shows the magnitude of the entries of a Matrix.
    Densifies."""

    def __init__(self, M,
                 legend_labels=None, title=None, x_label=None, y_label=None,
                 **mpl_kwargs):
        super().__init__(legend_labels, title, x_label, y_label, **mpl_kwargs)
        if not isinstance(M, Matrix):
            M = Matrix(M)
        if M.is_sparse():
            M = M.to_dense()
        if M.is_complex():
            M = Matrix(np.absolute(M.entries))
        self.M = M
        if "cmap" not in self._mpl_kwargs:
            self._mpl_kwargs["cmap"] = "gray_r"

    def plot_on(self, axes):
        pos = axes.matshow(self.M.entries, **self._mpl_kwargs)
        plt.gcf().colorbar(pos, ax=axes, fraction=0.046)
        super().plot_on(axes)


class SparsityPlot(MagnitudePlot):
    """A SparsityPlot shows the sparsity of a Matrix object.
    Optionally specify a tolerance to treat entries as zero. Densifies."""

    def __init__(self, M, tolerance=EPS,
                 legend_labels=None, title=None, x_label=None, y_label=None,
                 **mpl_kwargs):
        super().__init__(M, legend_labels, title, x_label, y_label,
                         **mpl_kwargs)
        self.M = Matrix(~np.isclose(self.M.entries, 0, atol=tolerance))

    def plot_on(self, axes):
        axes.matshow(self.M.entries, **self._mpl_kwargs)
        Plot.plot_on(self, axes)


class OrthogonalityPlot(MagnitudePlot):
    """An OrthogonalityPlot treats a Matrix as a set of column vectors,
    and shows the magnitude of each dot product by computing M^H M."""

    def __init__(self, M,
                 legend_labels=None, title=None, x_label=None, y_label=None,
                 **mpl_kwargs):
        super().__init__(M, legend_labels, title, x_label, y_label,
                         **mpl_kwargs)
        self.M = Matrix(
            np.abs((self.M.conjugate().transpose() @ self.M).entries))


def single(plot, **mpl_kwargs):
    """Show a single Plot on a new figure."""
    if not isinstance(plot, Plot):
        raise ValueError("Cannot plot {}.".format(str(plot)))

    figure = plt.figure(**mpl_kwargs)
    plot.plot_on(figure.gca())
    return figure


def tiled(plot_array, sup_title=None, sup_x_label=None, sup_y_label=None,
          **mpl_kwargs):
    """Tile the given plots in a grid.
    `plot_array` can be a single plot, a one-dimensional list,
    which will be rendered as a row of plots, or a two dimensional
    list, which will be rendered as a grid of plots."""
    if isinstance(plot_array, Plot):
        plot_array = [[plot_array]]
    if type(plot_array) is list or type(plot_array) is tuple:
        if type(plot_array[0]) is list or type(plot_array[0]) is tuple:
            nr = len(plot_array)
            nc = len(plot_array[0])
        else:
            nr = 1
            nc = len(plot_array)
            plot_array = [plot_array]
    else:
        raise ValueError(
            "Cannot interpret the plot array {}.".format(str(plot_array)))

    figure, axes_arr = plt.subplots(nr, nc, **mpl_kwargs)
    if len(axes_arr.shape) == 1:
        axes_arr = np.array([axes_arr], dtype=object)
    # Plot the super-titles and labels if specified
    if sup_title is not None:
        figure.suptitle(sup_title)
    if sup_x_label is not None:
        figure.supxlabel(sup_x_label)
    if sup_y_label is not None:
        figure.supylabel(sup_y_label)

    # Generate all plots
    for row in range(nr):
        for col in range(nc):
            plot_array[row][col].plot_on(axes_arr[row][col])

    return figure
