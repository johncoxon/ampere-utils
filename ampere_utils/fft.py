# encoding: utf-8
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import welch
from scipy.stats import f


def check_harmonics(frequency, period):
    """

    Parameters
    ----------
    frequency
    period

    Returns
    -------

    """
    indices = list(np.where(frequency == (1 / period))[0])

    for i in np.arange(2, 10):
        harmonic = (i / period)

        if (harmonic > frequency).all():
            break
        else:
            index = np.where(frequency == harmonic)[0]
            if len(index):
                indices.append(index[0])

    indices = np.sort(np.array(indices))
    return indices


def get_fft(dataset, use_fft=False, n_per_segment=None):
    """
    Compute the power spectral density (PSD) of the input data.

    Parameters
    ----------
    dataset : xarray.DataArray
        The data to compute the power spectral density for.
    use_fft : bool, optional, default False
        Set this True to use scipy.signal.rfft, if False uses Welch's method.
    n_per_segment
        Passed to the nperseg kwarg for scipy.signal.welch.

    Returns
    -------
    amplitude : np.ndarray of shape (n,)
        The amplitude of the PSD.
    frequency  : np.ndarray of shape (n,)
        The frequency of the PSD.
    fraction_interpolated : float
        The fraction of the data that was interpolated.
    """
    n = len(dataset)
    fraction_interpolated = np.isnan(dataset).data.sum() / n

    if n_per_segment:
        if n_per_segment % 1:
            raise ValueError("n_per_segment must be an integer.")
        else:
            n_per_segment = int(n_per_segment)
    else:
        n_per_segment = n

    if use_fft:
        amplitude = np.abs(rfft(dataset.interpolate_na(dim="time"))) / n
        frequency = rfftfreq(n_per_segment, d=1)
    else:
        frequency, psd = welch(dataset.interpolate_na(dim="time"), window="boxcar", nperseg=n_per_segment)
        amplitude = (psd / np.sum(psd))

    return amplitude, frequency, fraction_interpolated


def plot_fft(ax, amplitude, frequency, fraction_interpolated, peaks_on_legend=True, verbose=False, noise_colour="C3",
             significance_colour="black", ncols=3, **kwargs):
    """
    Plot the power spectral density of the data.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    amplitude : np.ndarray of shape (n,)
        The amplitude of the PSD from get_fft.
    frequency : np.ndarray of shape (n,)
        The frequency of the PSD from get_fft.
    fraction_interpolated : float
        The fraction of the data that was interpolated.
    peaks_on_legend : bool, optional, default True
        Whether to indicate the peaks on the legend. If set False, sets Verbose to True.
    verbose : bool, optional, default False
        Whether to print the peaks or not.
    noise_colour : string, optional, default "C3"
        The colour to plot the red noise line in. Defaults to red.
    significance_colour : string, optional, "black"
        The colour to plot the significance level in.
    ncols : int, optional, default 3
        The number of columns in the legend.
    **kwargs
        Passed to the underlying pyplot.plt call for the power spectral density.
    """
    # Construct expected red noise spectrum
    rho = 0.5  # Red noise lag-one autocorrelation
    m = len(frequency)
    l_h = ((1 - (rho ** 2)) /
           (1 - (2 * rho * np.cos((np.arange(0, m) * np.pi) / m)) + (rho ** 2)))

    if not peaks_on_legend:
        verbose = True

    frequency = frequency[1:]
    amplitude = amplitude[1:]
    l_h = l_h[1:]

    # Calculate the significance using F-testing
    fstat = f.ppf(0.997, 2 * 1.2, 100)
    significance = l_h * fstat

    significant_peaks = np.where(amplitude > (significance / np.sum(l_h)))[0]

    ax.plot(frequency, amplitude, **kwargs)
    ax.plot(frequency, significance / np.sum(l_h), label="99.7% confidence", color=significance_colour)
    ax.plot(frequency, l_h / np.sum(l_h), label="Red noise fit", color=noise_colour)

    ax.annotate(f"{fraction_interpolated * 100:.1f}% data interpolated", (1, 1), (-10, -10), "axes fraction", "offset points",
                ha="right", va="top", fontsize="x-small")

    if len(significant_peaks):
        for cnt, p in enumerate(significant_peaks):
            if peaks_on_legend:
                label = f"{1 / frequency[p]:.2f} days"
            else:
                label = None

            ax.axvline(frequency[p], alpha=0.2, color="black", label=label)
            if verbose:
                print(f"Periods bracketing significant peak {cnt}: {1 / frequency[p - 1]:8.3f}, "
                      f"{1 / frequency[p]:8.3f}, {1 / frequency[p + 1]:8.3f}")

        ax.legend(ncols=ncols, loc="lower left")
