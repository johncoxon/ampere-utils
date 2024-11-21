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

    Parameters
    ----------
    dataset
    use_fft
    n_per_segment

    Returns
    -------

    """
    n = len(dataset)
    n_interpolated = np.isnan(dataset).data.sum() / n

    if n_per_segment:
        if n_per_segment % 1:
            raise ValueError("n_per_segment must be an integer.")
        else:
            n_per_segment = int(n_per_segment)
    else:
        n_per_segment = n

    if use_fft:
        amplitude = np.abs(rfft(dataset.interpolate_na(dim="time"))) / n
    else:
        _, p = welch(dataset.interpolate_na(dim="time"), window="boxcar", nperseg=n_per_segment)
        amplitude = (p / np.sum(p))

    frequency = rfftfreq(n_per_segment, d=1)

    return amplitude, frequency, n_interpolated


def plot_fft(ax, amplitude, frequency, n_interpolated, verbose=False):
    """

    Parameters
    ----------
    ax
    amplitude
    frequency
    n_interpolated
    verbose

    Returns
    -------

    """
    # Construct expected red noise spectrum
    rho = 0.5  # Red noise lag-one autocorrelation
    m = len(frequency)
    l_h = ((1 - (rho ** 2)) /
           (1 - (2 * rho * np.cos((np.arange(0, m) * np.pi) / m)) + (rho ** 2)))

    frequency = frequency[1:]
    amplitude = amplitude[1:]
    l_h = l_h[1:]

    # Calculate the significance using F-testing
    fstat = f.ppf(0.997, 2 * 1.2, 100)
    significance = l_h * fstat

    significant_peaks = np.where(amplitude > (significance / np.sum(l_h)))[0]
    ax.plot(frequency, l_h / np.sum(l_h), '--', label='Red noise fit', color='C3')
    ax.plot(frequency, significance / np.sum(l_h), '--', label='99.7% confidence', color='C1')
    ax.plot(frequency, amplitude)

    ax.annotate(f"{n_interpolated * 100:.1f}% data interpolated", (1, 1), (-10, -10), "axes fraction", "offset points",
                ha="right", va="top", fontsize="x-small")

    if len(significant_peaks):
        for cnt, p in enumerate(significant_peaks):
            ax.axvline(frequency[p], ls=":", color=f"C{cnt}", label=f"{1 / frequency[p]:.2f}-day period")
            if verbose:
                print(f"Periods bracketing significant peak {cnt}: {1 / frequency[p - 1]:8.3f}, "
                      f"{1 / frequency[p]:8.3f}, {1 / frequency[p + 1]:8.3f}")

        ax.legend(ncols=2, loc="lower left")
