# encoding: utf-8
import numpy as np


def asymmetry_index(north, south):
    """Asymmetry index between a northern quantity and a southern quantity, ranging from -1 to 1."""
    return (north - south) / (north + south)


def colat_and_mlt(hemisphere="north"):
    """Returns colat and mlt for plotting with pcolormesh."""
    if hemisphere == "north":
        colat = np.linspace(1, 50, 50)
    else:
        colat = np.linspace(179, 130, 50)
    mlt = np.linspace(0, 23, 24)
    return colat, mlt
