import matplotlib.pyplot as plt
import numpy as np


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
    if out_file:  plt.savefig(out_file)
    if show:      plt.show()


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
    if out_file:  plt.savefig(out_file)
    if show:      plt.show()
