import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from ace_client import MAX_INPUT_RANGE, ADC_RES_BITS
from scipy.signal import find_peaks

MICRO = 1e6          # volts → micro-volts conversion factor
DB_REF = 1.0         # dB reference ( 1 V_rms ).  Change if you prefer dBVμ etc.

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
def plot_agg_histogram(raw_all, bins, runs, odr, filt,
                       out_file=None, show=False):
    # --- raw → integer ADC codes -------------------------------------------
    lsb       = MAX_INPUT_RANGE / (2 ** (ADC_RES_BITS - 1))
    codes     = np.round(raw_all / lsb).astype(int)

    n_codes   = codes.max() - codes.min() + 1
    grp_size  = max(1, int(np.ceil(n_codes / bins)))        # ≈ requested bins
    edges_c   = np.arange(codes.min() - 0.5,
                          codes.max() + 0.5 + grp_size,
                          grp_size)

    fig, ax   = plt.subplots()
    counts, _, _ = ax.hist(codes, bins=edges_c,
                           edgecolor='black', linewidth=0.5)

    # ---------- relabel x-ticks in µV --------------------------------------
    tick_c    = np.linspace(codes.min(), codes.max(), 6, dtype=int)
    tick_uV   = tick_c * lsb * MICRO
    ax.set_xticks(tick_c)
    ax.set_xticklabels([f"{v:.0f}" for v in tick_uV], rotation=45, ha='right')
    ax.set_xlabel('Voltage [µV]')
    ax.set_ylabel('Count')

    # ---------- Gaussian overlay (still computed in code units) ------------
    mu      = raw_all.mean()
    sigma   = raw_all.std(ddof=1)
    bin_w_V = grp_size * lsb
    centres = (edges_c[:-1] + edges_c[1:]) * 0.5 * lsb
    pdf     = (1 / (sigma * np.sqrt(2*np.pi))) * np.exp(-0.5*((centres - mu)/sigma)**2)
    ax.plot(centres / lsb, pdf * raw_all.size * bin_w_V,
            'r-', linewidth=1, label='Gaussian fit')

    # ---------- cosmetics ---------------------------------------------------
    ax.set_title(f"{runs}-run Noise Floor Histogram\n@ ODR {odr/1e6:.1f} MHz – {filt}")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, prune='both'))
    ax.legend()
    plt.tight_layout()

    if out_file:
        plt.savefig(out_file, dpi=300)
    if show:
        plt.show()


# Plot aggregated FFT/PSD with runs, ODR, and filter info in dB. Uses dB = 20·log10(mag / DB_REF) with DB_REF = 1 V_rms by default.
# param freqs: frequency bins (Hz)
# param mags: magnitude spectrum
# param runs: number of runs performed
# param odr: ODR rate in Hz
# param filt: filter name (string)
# param out_file: filename to save the figure (PNG)
# param show: if True, display the plot interactively
def plot_agg_fft(freqs, mags, runs, odr, filt, out_file=None, show=False):
    # 1) convert to dB
    mags_db = 20.0 * np.log10(np.maximum(mags, np.finfo(float).tiny) / DB_REF)

    # 2) start figure
    fig, ax = plt.subplots()
    ax.semilogx(freqs, mags_db, lw=0.8)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Magnitude [dBV]')
    ax.set_title(f"{runs}-run Noise Floor PSD @ {odr/1e6:.1f} MHz – {filt}")

    # 3) grid
    ax.grid(True, which='both', linestyle='--', linewidth=0.4)

    # 4) dynamic y‐limits (10% padding)
    y_min, y_max = mags_db.min(), mags_db.max()
    pad = 0.1 * (y_max - y_min)
    ax.set_ylim(y_min - pad, y_max + pad)

    # 5) find & annotate the two largest non-DC spurs
    """peaks, _ = find_peaks(mags_db, distance=5)
    # ignore the very DC‐adjacent bin
    valid = peaks[freqs[peaks] > odr*0.01]
    top2 = sorted(valid, key=lambda i: mags_db[i], reverse=True)[:2]
    for idx in top2:
        f = freqs[idx]
        db = mags_db[idx]
        ax.annotate(
            f"{f/1e3:.1f} kHz\n{db:.1f} dB",
            xy=(f, db),
            xytext=(0, 8),
            textcoords='offset points',
            ha='center',
            arrowprops=dict(arrowstyle='->', lw=0.8)
        )"""

    plt.tight_layout()
    if out_file:
        plt.savefig(out_file, dpi=300)
    if show:
        plt.show()
