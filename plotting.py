import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# Plot time-domain raw data
# param fs: sampling frequency (Hz)
# param raw: 1D array of voltage samples
# param out_file: filename to save the figure (PNG)
# param show: if True, display the plot interactively
def plot_raw(raw, out_file=None, show=False):
    plt.figure()
    plt.plot(raw)
    plt.xlabel('Sample Index')
    plt.ylabel('Voltage [V]')
    plt.title('Noise Floor (Raw)')
    plt.tight_layout()
    if out_file:
        plt.savefig(out_file)
    if show:
        plt.show()


# Plot histogram of raw data
# param raw: 1D array of voltage samples
# param bins: number of histogram bins
# param out_file: filename to save the figure (PNG)
# param show: if True, display the plot interactively
def plot_histogram(raw, bins=100, out_file=None, show=False):
    plt.figure()
    plt.hist(raw, bins=bins)
    plt.xlabel('Voltage [V]')
    plt.ylabel('Count')
    plt.title('Histogram')
    plt.tight_layout()
    if out_file:
        plt.savefig(out_file)
    if show:
        plt.show()


# Plot FFT of raw data
# param freqs: array of frequency bins (Hz)
# param spectrum: magnitude spectrum values
# param out_file: filename to save the figure (PNG)
# param show: if True, display the plot interactively
def plot_fft(freqs, spectrum, out_file=None, show=False):
    plt.figure()
    plt.plot(freqs, spectrum)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.title('FFT')
    plt.tight_layout()
    if out_file:
        plt.savefig(out_file)
    if show:
        plt.show()


# Plot settling transient
# param raw: 1D array of voltage samples
# param fs: sampling frequency (Hz)
# param out_file: filename to save the figure (PNG)
# param show: if True, display the plot interactively
def plot_settling(raw, fs, out_file=None, show=False):
    t = np.arange(raw.size) / fs
    plt.figure()
    plt.plot(t, raw)
    plt.axhline(np.mean(raw[int(0.9*raw.size):]), linestyle='--')
    plt.xlabel('Time [s]')
    plt.ylabel('Voltage [V]')
    plt.title('Settling Transient')
    plt.tight_layout()
    if out_file:
        plt.savefig(out_file)
    if show:
        plt.show()


# Plot frequency response
# param freqs: array of frequency bins (Hz)
# param gains: array of gain values (linear scale)
# param out_file: filename to save the figure (PNG)
# param show: if True, display the plot interactively
def plot_freq_response(freqs, gains, out_file=None, show=False):
    plt.figure()
    plt.semilogx(freqs, 20*np.log10(gains))
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Gain [dB]')
    plt.title('Frequency Response')
    plt.tight_layout()
    if out_file:
        plt.savefig(out_file)
    if show:
        plt.show()


def plot_dc_gain(actual_vs, adc_means, out_file=None, show=False):
    plt.figure()
    plt.plot(actual_vs, adc_means, 'o-')
    plt.xlabel('Actual Voltage (V)')
    plt.ylabel('ADC Measured Voltage (V)')
    plt.title('DC Gain and Offset')
    plt.tight_layout()
    if out_file:
        plt.savefig(out_file)
    if show:
        plt.show()

# New aggregated plotting functions
# Plot aggregated histogram with runs, ODR, and filter info
# param raw_all: concatenated array from all runs
# param bins: number of histogram bins
# param runs: number of runs performed
# param odr: ODR rate in Hz
# param filt: filter name (string)
# param out_file: filename to save the figure (PNG)
# param show: if True, display the plot interactively
def plot_agg_histogram(raw_all, bins, runs, odr, filt, out_file=None, show=False):
    fig, ax = plt.subplots()
    ax.hist(raw_all, bins=bins)
    ax.set_xlabel('Voltage [V]')
    ax.set_ylabel('Count')
    ax.set_title(f"{runs}-run Noise Floor Histogram @ {odr:.0f}Hz ({filt})")

    # fewer, rotated x-ticks so they don’t overlap
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, prune='both'))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    if out_file:
        plt.savefig(out_file)
    if show:
        plt.show()


# Plot aggregated FFT/PSD with runs, ODR, and filter info
# param freqs: frequency bins (Hz)
# param mags: magnitude spectrum
# param runs: number of runs performed
# param odr: ODR rate in Hz
# param filt: filter name (string)
# param out_file: filename to save the figure (PNG)
# param show: if True, display the plot interactively
def plot_agg_fft(freqs, mags, runs, odr, filt, out_file=None, show=False):
    plt.figure()
    plt.plot(freqs, mags)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.title(f"{runs}-run Noise Floor PSD @ {odr:.0f}Hz ({filt})")
    plt.tight_layout()
    if out_file:
        plt.savefig(out_file)
    if show:
        plt.show()
