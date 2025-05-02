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
def plot_settling(raw, fs, out_file=None, show=False, legend=True):
    """
    Plot one or many settling‑time captures.

    Parameters
    ----------
    raw       : np.ndarray | Iterable[np.ndarray]
        Single capture or list/tuple of captures (all same length).
    fs        : float
        Sampling frequency [Hz].
    out_file  : str | None      – PNG filename.
    show      : bool            – call plt.show().
    legend    : bool            – add legend when multiple runs supplied.
    """
    # --- normalise input to a list of 1‑D arrays --------------------------
    if isinstance(raw, np.ndarray):
        runs = [raw]
    elif isinstance(raw, Iterable):
        runs = list(raw)
        if not all(isinstance(r, np.ndarray) for r in runs):
            raise TypeError("Every element in raw must be a NumPy array")
    else:
        raise TypeError("raw must be an array or iterable of arrays")

    n = len(runs)
    t = np.arange(runs[0].size) / fs

    plt.figure()
    for idx, r in enumerate(runs, 1):
        α = 0.4 if n > 1 else 1.0          # light grey for individual runs
        plt.plot(t, r, color="grey", alpha=α,
                 label=f"Run {idx}" if legend and n > 1 else None)

    # --- aggregate --------------------------------------------------------
    if n > 1:
        mean_trace = np.mean(np.vstack(runs), axis=0)
        plt.plot(t, mean_trace, color="C0", linewidth=2,
                 label="Mean" if legend else None)
        ref_level = mean_trace[int(0.9 * mean_trace.size):].mean()
    else:
        ref_level = runs[0][int(0.9 * runs[0].size):].mean()

    plt.axhline(ref_level, linestyle="--", color="C1", linewidth=1,
                label="Final value" if legend else None)

    plt.xlabel("Time [s]")
    plt.ylabel("Voltage [V]")
    plt.title(f"Settling transient ({n} run{'s' if n > 1 else ''})")
    if legend and (n > 1):
        plt.legend()
    plt.tight_layout()

    if out_file:
        plt.savefig(out_file, dpi=150)
    if show:
        plt.show()
    plt.close()


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
# -------------------------------------------------------------------------
def plot_agg_histogram(raw_all, bins, runs, odr, filt,
                       out_file=None, show=False):
    """
    Aggregate noise-floor histogram, x-axis in µV but bin edges aligned to
    integer ADC codes → no “comb” effect between neighbour bars.
    """
    # --- constants ---------------------------------------------------------
    lsb      = MAX_INPUT_RANGE / (2 ** (ADC_RES_BITS - 1))
    lsb_uV   = lsb * MICRO

    # --- raw voltage → integer codes --------------------------------------
    codes    = np.round(raw_all / lsb).astype(int)

    # choose code-aligned bin edges (same logic as before)
    n_codes  = codes.max() - codes.min() + 1
    grp_size = max(1, int(np.ceil(n_codes / bins)))          # ≈ requested bins
    edges_c  = np.arange(codes.min() - 0.5,
                         codes.max() + 0.5 + grp_size,
                         grp_size)

    # scale edges into µV so histogram x-axis is really µV
    edges_uV = edges_c * lsb_uV

    # --- histogram --------------------------------------------------------
    fig, ax = plt.subplots()
    counts, _, _ = ax.hist(
        raw_all * MICRO,          # data in µV
        bins=edges_uV,            # µV edges aligned to codes
        edgecolor="black",
        linewidth=0.5,
    )

    # --- horizontal grid only --------------------------------------------
    ax.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.6)

    # --- axis labels ------------------------------------------------------
    ax.set_xlabel("Voltage [µV]")
    ax.set_ylabel("Count")

    # --- nice x-tick locations in µV -------------------------------------
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))

    # --- Gaussian overlay (all in µV) ------------------------------------
    mu_uV    = raw_all.mean() * MICRO
    sigma_uV = raw_all.std(ddof=1) * MICRO
    bin_w_uV = edges_uV[1] - edges_uV[0]
    centres  = (edges_uV[:-1] + edges_uV[1:]) * 0.5
    pdf      = (
        1.0 / (sigma_uV * np.sqrt(2 * np.pi))
        * np.exp(-0.5 * ((centres - mu_uV) / sigma_uV) ** 2)
    )
    ax.plot(
        centres,
        pdf * raw_all.size * bin_w_uV,
        "r-",
        linewidth=1,
        label="Gaussian fit",
    )

    # --- cosmetics --------------------------------------------------------
    ax.set_title(
        f"{runs}-run Noise Floor Histogram\n@ ODR {odr/1e6:.1f} MHz – {filt}"
    )
    ax.legend()
    plt.tight_layout()

    # --- save / show ------------------------------------------------------
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
# param xmin_hz: minimum frequency to plot (Hz)
def plot_agg_fft(freqs, mags, runs, odr, filt,
                 out_file=None, show=False, xmin_hz=5e3):
    mags_db = 20*np.log10(np.maximum(mags, np.finfo(float).tiny) / DB_REF)

    # ------ mask below xmin_hz ------
    keep          = freqs >= xmin_hz
    freqs_plot    = freqs[keep] / 1e3   # kHz axis
    mags_db_plot  = mags_db[keep]

    fig, ax = plt.subplots()
    ax.plot(freqs_plot, mags_db_plot, lw=0.8)
    ax.set_xlabel('Frequency [kHz]')
    ax.set_ylabel('Magnitude [dBV]')
    ax.set_title(f"{runs}-run Noise Floor PSD @ {odr/1e6:.1f} MHz – {filt}")

    ax.grid(True, which='both', linestyle='--', linewidth=0.4)

    # y-limit padding
    pad = 0.10 * (mags_db_plot.max() - mags_db_plot.min())
    ax.set_ylim(mags_db_plot.min() - pad, mags_db_plot.max() + pad)

    plt.tight_layout()
    if out_file:
        plt.savefig(out_file, dpi=300)
    if show:
        plt.show()
