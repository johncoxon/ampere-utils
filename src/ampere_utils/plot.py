# encoding: utf-8
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.quiver import Quiver
from matplotlib.ticker import FuncFormatter, MultipleLocator


def colourmap(vmin, vmax, white, cmap="RdBu_r"):
    """
    Returns a colour map which is white between -0.2 and 0.2, depending on input vmin and vmax.

    Parameters
    ----------
    vmin, vmax : float
        The minimum and maximum values of the colour bar in use.
    white : float, optional, default 0.2
        The number up to which the colour map should be white.
    cmap : str, optional, default "RdBu_r"
        The colourmap to use (set by default to the normal red/blue).

    Returns
    -------
    colourmap : matplotlib.colors.Colormap
        A colourmap that describes the extremes of the input colourmap as large values of positive/
            negative but has white in the middle between -0.2 and 0.2 (by default).
    """
    cmap = plt.get_cmap(cmap)

    if white:
        # Work out what percentage the -0.2, 0.2 range is in the given vmin, vmax range.
        number = int(np.round(128 * 2 * white / (float(vmax) - float(vmin))))
        n_half = 128 - number

        # Create a list which goes from dark red to light red, then goes white for the low values,
        #    then goes from light blue to dark blue.
        new_cmap = []
        new_cmap.extend(np.linspace(0, 0.49, n_half))
        new_cmap.extend(np.ones(number * 2) * 0.5)
        new_cmap.extend(np.linspace(0.51, 1, n_half))

        # Return a colourmap from that list.
        cmap = ListedColormap(cmap(new_cmap))

    return cmap


def configure_polar_plot(ax, rmax, colat_grid_spacing=10, mlt=True, theta_range=None, theta_zero_location=0,
                         theta_reversed=False, coordinate_labels=False):
    """
    Configures a polar plot to appear on the page correctly.

    Parameters
    ----------
    ax : matplotlib.projections.polar.PolarAxes
    rmax : float
    colat_grid_spacing : float, optional, default 10
    mlt : bool, optional, default True
    theta_range : listlike, optional, default None
    theta_zero_location : float, optional, default 0
    theta_reversed : bool, optional, default False
    coordinate_labels : bool, optional, default False
    """
    ax.set_rmin(0.0)
    ax.set_rmax(rmax)
    ax.yaxis.set_major_locator(MultipleLocator(colat_grid_spacing))

    # Check if the plot is a quiver plot.
    is_quiver = np.array([isinstance(c, Quiver) for c in ax.axes.get_children()]).any()

    # If it is a quiver, check whether theta_zero_location is consistent and throw an error if it isn't.
    if is_quiver:
        if ax.get_theta_offset() != np.radians(270 - theta_zero_location):
            raise ValueError("polar_quiver and configure_polar_plot should have the same theta_zero_location.")
        if theta_reversed:
            if ax.get_theta_direction() == 1:
                raise ValueError("Reversing theta direction after a quiver plot will cause bad behaviour.")
    else:
        ax.set_theta_offset(np.radians(270 - theta_zero_location))

    if theta_range is not None:
        ax.set_thetamin(theta_range[0])
        ax.set_thetamax(theta_range[1])

    if mlt:
        ax.xaxis.set_major_formatter(format_mlt())
        ax.xaxis.set_major_locator(MultipleLocator(np.pi / 2))

    if not coordinate_labels:
        ax.set(yticklabels=[])

    if theta_reversed:
        ax.set_theta_direction(-1)
    ax.grid(True)


def draw_labels(axes, xy=(0, 1), xytext=(10, -10), transpose=False, size="x-large", **kwargs):
    """Draws labels on each subplot in an array."""
    if transpose:
        shape = axes.T.shape
    else:
        shape = axes.shape

    arange = np.arange(0, np.prod(shape))
    labels = []
    string = 'abcdefghijklmnopqrstuvwxyz'

    for i in arange:
        labels.append(string[i])

    label_list = np.reshape(np.array(labels), shape)

    for row_cnt, row in enumerate(axes):
        try:
            for col_cnt, ax in enumerate(row):
                try:
                    if transpose:
                        label = label_list[col_cnt, row_cnt]
                    else:
                        label = label_list[row_cnt, col_cnt]
                    ax.annotate(label, xy, xytext=xytext,
                                xycoords="axes fraction", textcoords="offset points", va="top",
                                ha="left", size=size, **kwargs)
                except AttributeError:
                    pass
        except TypeError:
            try:
                row.annotate(label_list[row_cnt], xy, xytext=xytext, xycoords="axes fraction",
                             textcoords="offset points", va="top", ha="left", size=size, **kwargs)
            except AttributeError:
                pass


def format_mlt():
    """Return MLT in hours rather than a number of degrees when drawing axis labels."""

    # noinspection PyUnusedLocal
    def formatter_function(y, pos):
        hours = y * (12 / np.pi)
        if hours == 24:
            return ""
        else:
            if hours < 0:
                hours += 24
            return f"{hours:.0f}"

    return FuncFormatter(formatter_function)


def format_north_colatitude():
    """Return colatitudes for the Northern Hemisphere."""

    # noinspection PyUnusedLocal
    def formatter_function(y, pos):
        return f"{np.absolute(y):.0f}°"

    return FuncFormatter(formatter_function)


def format_south_colatitude():
    """Return colatitudes for the Southern Hemisphere."""

    # noinspection PyUnusedLocal
    def formatter_function(y, pos):
        return f"{180 - np.absolute(y):.0f}°"

    return FuncFormatter(formatter_function)


def mlt_from_j_and_colat(j, colat, mlt, ax, reverse_x_axis=False, **kwargs):
    """
    Plot j against colatitude for a given MLT.

    Parameters
    ----------
    j : np.ndarray
        The array containing the current for the relevant MLT slice.
    colat : np.ndarray
        The array containing the colatitude.
    mlt : int
        The MLT you want to plot.
    ax : matplotlib.axes.Axes
        The ax object on which you want to plot.
    reverse_x_axis : bool, optional, default False
        Flip the x-axis (for having the pole in the middle of two-column figs).

    Other kwargs will be passed to the first ax.plot() call.
    """
    ax.plot(colat, j, label="Data", **kwargs)

    # Set variables for drawing annotations and note the hour of MLT.
    xy = {"left": [0, 1], "right": [1, 1]}
    offset = {"left": [6, -6], "right": [-6, -6]}

    if colat[0] > 90:
        text_index = "left" if reverse_x_axis else "right"
    else:
        text_index = "right" if reverse_x_axis else "left"

    ax.annotate(f"{mlt}", xy=xy[text_index], xycoords="axes fraction",
                xytext=offset[text_index], textcoords="offset points",
                ha=text_index, va="top")

    ax.axhspan(-0.2, 0.2, alpha=0.1, edgecolor=None)

    ax.yaxis.set_minor_locator(MultipleLocator(0.2))

    if reverse_x_axis:
        xlim = ax.get_xlim()
        ax.set_xlim(xlim[-1], xlim[0])
    ax.set_ylim(-1.1, 1.1)

    return ax


def polar_plot(mlt, colat, data, hemisphere, ax, title=None, cmap="RdBu_r", vmin=None, vmax=None, white=None, rmax=50,
               colat_grid_spacing=10, theta_range=None, coordinate_labels=True, longitude=False):
    """
    Plot current density data on a polar graph with a colour scale.

    Parameters
    ----------
    mlt : numpy.ndarray
        MLT values as an array.
    colat : numpy.ndarray
        Colatitudes as an array.
    data : numpy.ndarray
        Data (e.g. currents) as an array.
    hemisphere : basestring
        The hemisphere as a string.
    ax : matplotlib.projections.polar.PolarAxes
        Axis on which you want to plot data.
    title : str, optional, default None
        Set this to a string to entitle the plot.
    cmap : basestring or matplotlib.colors.Colormap, optional, default "RdBu_r"
        The colourmap to use for the plot.
    vmin, vmax : float, optional, default None
        The minimum and maximum of the colour bar.
    white : float, optional, default None
        The value of data beneath which the colour scale will be white.
    rmax : float, optional, default None
        Set the radial extent of the plot.
    colat_grid_spacing : int
        The gridline spacing in colatitude.
    theta_range : listlike of two floats
        Maximum and minimum theta of the polar plot.
    coordinate_labels : bool, optional, default False
        Set this to suppress the colatitude and MLT tick labels.
    longitude : bool, optional, default False
        Set this True to pass longitudes to the routine instead of MLT.

    Returns
    -------
    mesh : matplotlib.collections.QuadMesh
        The mesh returned by pcolormesh.
    """
    if hemisphere == "south":
        colat_plot = 180.0 - colat
        ax.yaxis.set_major_formatter(format_south_colatitude())
    elif hemisphere == "north":
        colat_plot = colat
        ax.yaxis.set_major_formatter(format_north_colatitude())
    else:
        raise ValueError("Hemisphere set to invalid value: {}".format(hemisphere))

    if title:
        ax.set_title(title)

    # Mask values of current which are too low to plot.
    if white:
        data_ma = np.ma.array(data)
        data_masked = np.ma.masked_where(((data_ma < white) & (data_ma > (-white))), data_ma)
    else:
        data_masked = data

    if data_masked.shape[0] != colat_plot.shape[0]:
        data_masked = data_masked.T

    if not vmin and not vmax:
        vmax = np.max(np.absolute(data_masked))
        vmin = -vmax

    if longitude:
        lon_plot = np.radians(mlt)
    else:
        lon_plot = mlt * np.pi / 12

    mesh = ax.pcolormesh(lon_plot, colat_plot, data_masked, vmin=vmin, vmax=vmax,
                         cmap=colourmap(vmin, vmax, white=white, cmap=cmap), shading="auto")

    configure_polar_plot(ax, rmax, colat_grid_spacing=colat_grid_spacing, theta_range=theta_range,
                         coordinate_labels=coordinate_labels, mlt=(not longitude))

    return mesh


def polar_quiver(ax, colat, mlt, north, east, hemisphere, theta_zero_location=0, theta_reversed=False,
                 longitude=False, **kwargs):
    """
    Plot a quiver plot that works on a polar axis.

    Parameters
    ----------
    ax : matplotlib.projections.polar.PolarAxes
    colat : np.ndarray
        Colatitudes at which to plot arrows, in degrees.
    mlt : np.ndarray
        MLTs at which to plot arrows.
    north, east : np.ndarray
        The northward and eastward vector components.
    hemisphere : basestring
    theta_zero_location : float, optional, default 0
        The longitude which is pointing down the page, in degrees. (This is the same quantity as that set by
        configure_polar_plot, and should not be different to that quantity.)
    theta_reversed : bool, optional, default False
        Set this to reverse the theta direction (for non-Glass Earth projections).
    longitude : bool, optional, default False
        Set this True to pass longitudes to the routine instead of MLT.

    Returns
    -------
    q : matplotlib.quiver.Quiver
    """
    if longitude:
        lon_plot = np.radians(mlt)
    else:
        lon_plot = mlt * np.pi / 12
        longitude = mlt * 15

    if theta_reversed:
        longitude_corrected = longitude * -1
        east_corrected = east * -1
        ax.set_theta_direction(-1)
    else:
        longitude_corrected = longitude
        east_corrected = east

    if hemisphere == "north":
        local_angle = np.degrees(np.arctan2(north, east_corrected)) + longitude_corrected - theta_zero_location
        colat_plot = colat
        ax.yaxis.set_major_formatter(format_north_colatitude())
    else:
        local_angle = np.degrees(np.arctan2(-north, east_corrected)) + longitude_corrected - theta_zero_location
        colat_plot = 180 - colat
        ax.yaxis.set_major_formatter(format_south_colatitude())

    q = ax.quiver(lon_plot, colat_plot, north, east, angles=local_angle, **kwargs)
    ax.set_theta_offset(np.radians(270 - theta_zero_location))

    return q


def save_figure(fig, *args, title=None, **kwargs):
    """Saves and closes a figure."""
    if title:
        fig.suptitle(title)

    fig.savefig(*args, **kwargs)
    plt.close(fig)
    del fig
