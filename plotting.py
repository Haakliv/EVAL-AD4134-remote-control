import matplotlib.pyplot as plt
import numpy as np
from ace_client import MAX_INPUT_RANGE, ADC_RES_BITS

MICRO = 1e6          # volts → micro-volts conversion factor
DB_REF = 1.0         # dB reference ( 1 V_rms ).  Change if you prefer dBVμ etc.

def plot_settling_time(
    raw_runs: list[np.ndarray],
    start_idxs: list[int],
    end_idxs:   list[int],
    Ts_us:      float,
    time_vec:   np.ndarray,
    mean_seg:   np.ndarray,
    mean_delta: float,
    filt:       str,
    odr:        float,
    frequency:  float,
    amplitude:  float,
    runs:       int,
    out_file:   str = None,
    show:       bool = False,
):
    """
    Plots per-run traces in gray, then overlays the mean settling trace.
    Matches signature style of your other plot_* functions.

    :param raw_runs:  list of raw sample arrays
    :param start_idxs: list of start-indices per run
    :param end_idxs:   list of end-indices per run
    :param Ts_us:      sampling period in µs
    :param time_vec:   truncated time vector for mean trace (µs)
    :param mean_seg:   truncated mean trace
    :param mean_delta: mean settling time (µs)
    :param odr:        sampling rate (Hz)
    :param frequency:  stimulus frequency (Hz)
    :param amplitude:  stimulus amplitude (Vpp)
    :param runs:       number of runs
    :param out_file:   filename to save the figure (PNG)
    :param show:       if True, display the plot interactively
    """
    pad = 10  # two samples before and after

    plt.figure(figsize=(10, 6))

    # individual runs in gray
    for raw, s_idx, e_idx in zip(raw_runs, start_idxs, end_idxs):
        start = max(0, s_idx)
        end   = min(len(raw), e_idx + pad)
        t     = (np.arange(start, end) * Ts_us) - (s_idx * Ts_us)
        seg   = raw[start:end]
        plt.plot(t, seg, color='gray', alpha=0.2)

    # truncate out the very first sample of the mean trace
    t_mean = time_vec[1:]
    seg_mean = mean_seg[1:]

    # mean trace overlay
    plt.plot(t_mean, seg_mean, label='Mean Settling', linewidth=2)
    plt.axvline(0, linestyle=':', label='Mean start', linewidth=2)
    plt.axvline(mean_delta, linestyle='--', label='Mean end', linewidth=2)
    plt.axvspan(0, mean_delta, alpha=0.2, label='Mean window')
    xmin, xmax = t_mean[0], t_mean[-1]
    plt.xlim(xmin, xmax)

    # labels & ticks
    plt.xlabel('Time relative to edge (µs)')
    plt.ylabel('Voltage')

    # x‐ticks every 0.8µs from the truncated window bounds
    min_t, max_t = t_mean[0], t_mean[-1]
    plt.xticks(np.arange(min_t, max_t + 1e-9, 0.8))

    # title with all parameters
    plt.title(
        f"{runs}-run Settling Transient\n"
        f"@ {frequency:.1f}Hz, {amplitude:.2f}Vpp, {filt} - ODR {odr/1e6:.2f}MHz"
    )

    plt.grid(axis='x', which='major', alpha=0.3)
    plt.legend(ncol=2, fontsize='small')
    plt.tight_layout()

    if out_file:
        plt.savefig(out_file, dpi=300)
    if show:
        plt.show()

# Plot frequency response
def plot_freq_response(freqs,
                       gains,
                       runs: int,
                       amplitude_vpp: float,
                       out_file: str | None = None,
                       show: bool = False):
    """
    Frequency-response plot (no error overlay).

    Parameters
    ----------
    freqs, gains    : sweep vectors (linear gain)
    runs            : how many sweeps were averaged
    amplitude_vpp   : differential drive (Vpp)
    out_file, show  : as before
    """
    plt.figure()
    gdB = 20 * np.log10(gains)
    plt.semilogx(freqs, gdB, label="Mean")

    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Gain [dB]")
    plt.title(
        f"{runs}-run Frequency Response  "
        f"@ {amplitude_vpp*2:.2f} Vpp (normalised)"
    )
    plt.ylim((-25, 6))
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    plt.legend()

    if out_file:
        plt.savefig(out_file, dpi=300)
    if show:
        plt.show()

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
        f"{runs}-run Noise Floor Histogram\n@ ODR {odr/1e6:.1f} MHz - {filt}"
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
    ax.set_title(f"{runs}-run Noise Floor PSD @ {odr/1e6:.1f} MHz - {filt}")

    ax.grid(True, which='both', linestyle='--', linewidth=0.4)

    # y-limit padding
    pad = 0.10 * (mags_db_plot.max() - mags_db_plot.min())
    ax.set_ylim(mags_db_plot.min() - pad, mags_db_plot.max() + pad)

    plt.tight_layout()
    if out_file:
        plt.savefig(out_file, dpi=300)
    if show:
        plt.show()

def plot_fft_with_metrics(
    freqs,
    mag,
    fs,
    sfdr,
    thd,
    sinad,
    enob,
    runs: int,
    tone_freq: float,
    amplitude_vpp: float,
    filt: str,
    out_file: str | None = None,
    show: bool = False,
    xlim: tuple[float, float] | None = None,
):
    """
    FFT spectrum with SFDR / THD / SINAD / ENOB metrics.
    Title style is consistent with the other plot_* helpers.

    Parameters
    ----------
    freqs, mag     : FFT result vectors (Hz, Vrms/bin)
    fs             : sample rate (Hz)
    sfdr, thd, sinad, enob : already-computed figures
    runs           : number of averaged captures
    tone_freq      : stimulus tone frequency (Hz)
    amplitude_vpp  : differential drive level (Vpp) – shown for context
    filt           : filter name (e.g. 'Sinc6')
    out_file, show : as usual
    xlim           : (xmin_kHz, xmax_kHz) for the plot; values in kHz
    """
    # dB scale, referenced to 1 Vrms
    mag_db = 20.0 * np.log10(np.maximum(mag, np.finfo(float).tiny))
    f_khz  = freqs / 1e3          # x-axis in kHz

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(f_khz, mag_db, lw=0.9, label="Averaged spectrum")

    ax.set_xlabel("Frequency [kHz]")
    ax.set_ylabel("Magnitude [dBV]")

    title_main = (
        f"{runs}-run FFT Spectrum\n"
        f"@ {tone_freq:.1f} Hz, {amplitude_vpp:.2f} Vpp, "
        f"{filt} - ODR {fs/1e6:.2f} MHz"
    )
    title_metrics = (
        f"SFDR={sfdr:.2f} dB, THD={thd:.2f} dB, "
        f"SINAD={sinad:.2f} dB, ENOB={enob:.2f} bits"
    )
    ax.set_title(f"{title_main}\n{title_metrics}")

    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(loc="best", fontsize="small")

    if xlim is not None:
        ax.set_xlim(*xlim)        # values already in kHz

    plt.tight_layout()
    if out_file:
        plt.savefig(out_file, dpi=300)
    if show:
        plt.show()
    plt.close(fig)

def plot_dc_linearity_summary(
    actual_v_run: np.ndarray,
    adc_v_run: np.ndarray,
    fit_line_run: np.ndarray,
    inl_lsb_run: np.ndarray,
    avg_gain: float,
    avg_offset_uV: float,
    avg_max_inl_ppm: float,
    avg_rms_inl_lsb: float,
    runs: int,
    amplitude_vpp: float,
    steps: int,
    out_file: str | None = None,
    show: bool = False,
):
    # --- Style constants --------------------------------------------------
    lw = 1.5           # one place, one value
    ms = 4             # marker size (points)
    c_main, c_ideal, c_error = "C0", "gray", "black"

    fig, axs = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True,
        gridspec_kw={"hspace": 0.12}, constrained_layout=True
    )

    # X-axis limits --------------------------------------------------------
    if actual_v_run.size:
        xmin, xmax = actual_v_run.min(), actual_v_run.max()
        pad = 0.05 * (xmax - xmin) if xmax > xmin else 0.1
        xlim = (xmin - pad, xmax + pad)
    else:
        xlim = (-1, 1)

    # ---------------- Transfer plot --------------------------------------
    ax1 = axs[0]
    ax1.plot(xlim, xlim, c=c_ideal, ls="--", lw=lw, label="Ideal (y=x)")

    if actual_v_run.size:
        delta   = adc_v_run - actual_v_run

        ax1.plot(actual_v_run, fit_line_run,
                 c=c_main, lw=lw, label="Best-fit")

    ax1.set_ylabel("ADC Output [V]")
    ax1.grid(ls=":", lw=0.5, alpha=0.7)
    ax1.legend(fontsize="small")
    ax1.set_xlim(xlim)
    ax1.tick_params(axis="x", labelbottom=False)

    # ---------------- INL plot -------------------------------------------
    ax2 = axs[1]
    if inl_lsb_run.size:
        ax2.plot(
            actual_v_run, inl_lsb_run,
            marker="o", linestyle="None",  # <-- same marker as ax1
            ms=ms, lw=lw, c=c_main, label="INL"
        )

    ax2.axhline(0, c=c_ideal, ls="--", lw=lw)
    ax2.set_xlabel("Actual Input [V]")
    ax2.set_ylabel("INL [LSB]")
    ax2.grid(ls=":", lw=0.5, alpha=0.7)
    ax2.legend(fontsize="small")
    ax2.set_xlim(xlim)

    # ---------------- Title ----------------------------------------------
    fig.suptitle(
        f"{runs}-Run DC Linearity ({steps} steps, ±{amplitude_vpp/2:.2f} V)\n",
        fontsize=12
    )

    # ---------------- Save / show ----------------------------------------
    if out_file:
        plt.savefig(out_file, dpi=300)
    if show:
        plt.show()
    plt.close(fig)
