import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from ace_client import MAX_INPUT_RANGE, ADC_RES_BITS
from scipy.signal import find_peaks
from collections.abc import Iterable
from processing import compute_settling_time
import math

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

def plot_settling(
    raws: list[np.ndarray],      # Raw data arrays (one per run)
    fs: float,                   # ADC sampling rate [Hz]
    order: int,                  # Decimation filter order
    odr: float,                  # Output data rate [Hz]
    settle_start_us: float,         # Settle start time [µs]
    settle_end_us: float,           # Settle end time [µs]
    hold_samp: int,              # Hold-window length (samples) – for legend text
    tol_uV: float | None = None, # Band value retained in signature; omitted in plot
    pre_us: float = 3.0,         # Time before edge to show baseline [µs]
    post_gd: float = 0.7,        # Extent after edge as multiple of group delay
    show_sample_grid: bool = True,
    out_file: str | None = None,
    show: bool = False,
):
    """Single-panel settling plot showing only mean results
    • t = 0 at first displayed sample; rising-edge at t = pre_us.
    • Only the **mean** Settle Start and End are marked.
    """
    Ts_us = 1e6 / fs
    gd_us = (order + 1) / (2 * odr) * 1e6
    post_us = post_gd * gd_us
    
    # Calculate aligned data with the SAME edge detection approach for both raw and mean
    aligned_data = []
    edges = []
    
    # Step 1: Detect edges using same approach for all traces
    for r in raws:
        diff = np.diff(r)
        if not np.any(diff):
            edge_idx = 0
        elif np.all(diff <= 0):
            edge_idx = np.argmin(diff) + 1
        else:
            edge_idx = np.argmax(diff) + 1
        edges.append(edge_idx)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Step 2: Calculate number of samples to include before/after edge
    pre_samples = int(pre_us / Ts_us)
    post_samples = int(post_us / Ts_us)
    
    # Step 3: Create time axis once - this will be used for all plots
    t = np.arange(-pre_samples, post_samples + 1) * Ts_us
    
    # Step 4: Plot raw traces with exact same alignment
    for i, (r, edge) in enumerate(zip(raws, edges)):
        # Extract window around edge
        start_idx = max(0, edge - pre_samples)
        end_idx = min(r.size, edge + post_samples + 1)
        
        # Get the actual slice
        slice_data = r[start_idx:end_idx]
        
        # Calculate offset if we couldn't get full pre window
        offset = pre_samples - (edge - start_idx)
        
        # Create time values for this slice
        slice_t = t[offset:offset+len(slice_data)]
        
        # Plot raw trace
        ax.plot(slice_t, slice_data, color='0.75', alpha=0.7,
                label='Raw (each run)' if i == 0 else None)
        
        # Store aligned data for mean calculation
        aligned_data.append((slice_t, slice_data))
    
    # Step 5: Calculate mean trace - use the same windowing/alignment approach
    # First, create a common time grid
    t_common = np.arange(-pre_samples, post_samples + 1) * Ts_us
    mean_data = np.zeros_like(t_common)
    count = np.zeros_like(t_common)
    
    # Add each aligned trace to the mean
    for t_slice, data_slice in aligned_data:
        # Find where this slice maps to common time grid
        for i, t_val in enumerate(t_slice):
            idx = np.argmin(np.abs(t_common - t_val))
            mean_data[idx] += data_slice[i]
            count[idx] += 1
    
    # Divide by count to get mean (avoiding div by zero)
    mean_data = np.where(count > 0, mean_data / count, np.nan)
    
    # Step 6: Plot mean trace
    ax.plot(t_common, mean_data, color='C0', lw=2, label='Mean')

    #start time line
    ax.axvline(t_common[0], linestyle='--', color='C1', lw=1.5,)
    #0 time line
    ax.axvline(0, linestyle='--', color='C2', lw=1.5,)
    # middle of ramp time line
    

    # setle time lines
    ax.axvline(-2.4+settle_start_us, linestyle='--', color='k', lw=1.5,
               label=f'Mean Settle Start ({settle_start_us:.2f} µs)')
    ax.axvline(-2.4+settle_end_us, linestyle='--', color='C3', lw=1.5,
               label=f'Mean Settle End ({settle_end_us:.2f} µs)')
    
    # Sample grid
    if show_sample_grid:
        for n in range(-pre_samples, post_samples + 1):
            ax.axvline(n * Ts_us, color='0.88', lw=0.5, zorder=-1)
    
    # Cosmetics
    ax.set_xlim(t_common[0], t_common[-1])
    ax.set_ylim(-4.5, 4.5)
    ax.set_xlabel('Time [µs]')
    ax.set_ylabel('Voltage [V]')
    ax.set_title(f'Settling transient ({len(raws)} runs)')
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.8))  # Use Ts_us for tick spacing
    ax.grid(True, axis='x', linestyle=':', alpha=0.5)
    ax.legend(loc='upper left')
    
    if out_file:
        fig.savefig(out_file, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)

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
