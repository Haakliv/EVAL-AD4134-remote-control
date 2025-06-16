import matplotlib.pyplot as plt
import numpy as np
from common import MAX_INPUT_RANGE, ADC_RES_BITS, MICRO, KILO
from matplotlib.ticker import MaxNLocator

def plot_settling_time(
    raw_runs: list[np.ndarray],
    start_idxs: list[int],
    end_idxs:   list[int],
    ts_us:      float,
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
    pad = 10

    plt.figure(figsize=(10, 6))

    for raw, s_idx, e_idx in zip(raw_runs, start_idxs, end_idxs):
        start = max(0, s_idx)
        end   = min(len(raw), e_idx + pad)
        t     = (np.arange(start, end) * ts_us) - (s_idx * ts_us)
        seg   = raw[start:end]
        plt.plot(t, seg, color='gray', alpha=0.2)

    # Truncate out the very first sample of the mean trace
    t_mean = time_vec[1:]
    seg_mean = mean_seg[1:]

    plt.plot(t_mean, seg_mean, label='Mean Settling', linewidth=2)
    plt.axvline(0, linestyle=':', label='Mean start', linewidth=2)
    plt.axvline(mean_delta, linestyle='--', label='Mean end', linewidth=2)
    plt.axvspan(0, mean_delta, alpha=0.2, label='Mean window')
    xmin, xmax = t_mean[0], t_mean[-1]
    plt.xlim(xmin, xmax)

    plt.xlabel('Time relative to edge (µs)')
    plt.ylabel('Voltage')

    min_t, max_t = t_mean[0], t_mean[-1]
    plt.xticks(np.arange(min_t, max_t + 1e-9, 0.8))

    plt.title(
        f"{runs}-run Settling Transient\n"
        f"@ {frequency:.1f}Hz, {amplitude:.2f}Vpp, {filt} - ODR {odr/MICRO:.2f}MHz"
    )

    plt.grid(axis='x', which='major', alpha=0.3)
    plt.legend(ncol=2, fontsize='small')
    plt.tight_layout()

    if out_file:
        plt.savefig(out_file, dpi=300)
    if show:
        plt.show()

def plot_freq_response(
    freqs,
    gains,
    runs: int,
    amplitude_vpp: float,
    out_file: str | None = None,
    show: bool = False
):
    
    plt.figure()
    gain_db = 20 * np.log10(gains)
    plt.semilogx(freqs, gain_db, label="Mean")

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

def plot_agg_histogram(
    raw_all,
    bins,
    runs,
    odr,
    filt,
    out_file=None,
    show=False
):
    lsb      = MAX_INPUT_RANGE / (2 ** (ADC_RES_BITS - 1))
    lsb_u_v   = lsb * MICRO

    codes    = np.round(raw_all / lsb).astype(int)

    # Choose code-aligned bin edges
    n_codes  = codes.max() - codes.min() + 1
    group_size = max(1, int(np.ceil(n_codes / bins)))
    edges_c  = np.arange(codes.min() - 0.5,
                         codes.max() + 0.5 + group_size,
                         group_size)

    edges_u_v = edges_c * lsb_u_v

    _, ax = plt.subplots()
    _, _, _ = ax.hist(
        raw_all * MICRO,
        bins=edges_u_v,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.6)

    ax.set_xlabel("Voltage [µV]")
    ax.set_ylabel("Count")

    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))

    # Gaussian overlay
    mu_u_v    = raw_all.mean() * MICRO
    sigma_u_v = raw_all.std(ddof=1) * MICRO
    bin_w_u_v = edges_u_v[1] - edges_u_v[0]
    centres  = (edges_u_v[:-1] + edges_u_v[1:]) * 0.5
    pdf      = (
        1.0 / (sigma_u_v * np.sqrt(2 * np.pi))
        * np.exp(-0.5 * ((centres - mu_u_v) / sigma_u_v) ** 2)
    )
    ax.plot(
        centres,
        pdf * raw_all.size * bin_w_u_v,
        "r-",
        linewidth=1,
        label="Gaussian fit",
    )

    ax.set_title(
        f"{runs}-run Noise Floor Histogram\n@ ODR {odr/MICRO:.1f} MHz - {filt}"
    )
    ax.legend()
    plt.tight_layout()

    if out_file:
        plt.savefig(out_file, dpi=300)
    if show:
        plt.show()

def plot_agg_fft(
    freqs,
    magnitude,
    runs,
    odr,
    filt,
    out_file=None,
    show=False,
    xmin_hz=5e3
):
    magnitudes_db = 20*np.log10(np.maximum(magnitude, np.finfo(float).tiny))

    keep               = freqs >= xmin_hz
    freqs_plot         = freqs[keep] / KILO
    magnitudes_db_plot = magnitudes_db[keep]

    _, ax = plt.subplots()
    ax.plot(freqs_plot, magnitudes_db_plot, lw=0.8)
    ax.set_xlabel('Frequency [kHz]')
    ax.set_ylabel('Magnitude [dBV]')
    ax.set_title(f"{runs}-run Noise Floor PSD @ {odr/MICRO:.1f} MHz - {filt}")

    ax.grid(True, which='both', linestyle='--', linewidth=0.4)

    # y-limit padding
    pad = 0.10 * (magnitudes_db_plot.max() - magnitudes_db_plot.min())
    ax.set_ylim(magnitudes_db_plot.min() - pad, magnitudes_db_plot.max() + pad)

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
    # dB scale, referenced to 1 Vrms
    magnitude_db = 20.0 * np.log10(np.maximum(mag, np.finfo(float).tiny))
    f_khz  = freqs / KILO

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(f_khz, magnitude_db, lw=0.9, label="Averaged spectrum")

    ax.set_xlabel("Frequency [kHz]")
    ax.set_ylabel("Magnitude [dBV]")

    title_main = (
        f"{runs}-run FFT Spectrum\n"
        f"@ {tone_freq:.1f} Hz, {amplitude_vpp:.2f} Vpp, "
        f"{filt} - ODR {fs/MICRO:.2f} MHz"
    )
    title_metrics = (
        f"SFDR={sfdr:.2f} dB, THD={thd:.2f} dB, "
        f"SINAD={sinad:.2f} dB, ENOB={enob:.2f} bits"
    )
    ax.set_title(f"{title_main}\n{title_metrics}")

    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(loc="best", fontsize="small")

    if xlim is not None:
        ax.set_xlim(*xlim)

    plt.tight_layout()
    if out_file:
        plt.savefig(out_file, dpi=300)
    if show:
        plt.show()
    plt.close(fig)

def plot_dc_linearity_summary(
    actual_v_run: np.ndarray,
    fit_line_run: np.ndarray,
    inl_lsb_run: np.ndarray,
    runs: int,
    amplitude_vpp: float,
    steps: int,
    out_file: str | None = None,
    show: bool = False,
):
    line_width = 1.5
    marker_size = 4
    c_main, c_ideal = "C0", "gray"

    fig, axs = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True,
        gridspec_kw={"hspace": 0.12}, constrained_layout=True
    )

    if actual_v_run.size:
        xmin, xmax = actual_v_run.min(), actual_v_run.max()
        pad = 0.05 * (xmax - xmin) if xmax > xmin else 0.1
        xlim = (xmin - pad, xmax + pad)
    else:
        xlim = (-1, 1)

    ax1 = axs[0]
    ax1.plot(xlim, xlim, c=c_ideal, ls="--", lw=line_width, label="Ideal (y=x)")

    if actual_v_run.size:
        ax1.plot(actual_v_run, fit_line_run,
                 c=c_main, lw=line_width, label="Best-fit")

    ax1.set_ylabel("ADC Output [V]")
    ax1.grid(ls=":", lw=0.5, alpha=0.7)
    ax1.legend(fontsize="small")
    ax1.set_xlim(xlim)
    ax1.tick_params(axis="x", labelbottom=False)

    ax2 = axs[1]
    if inl_lsb_run.size:
        ax2.plot(
            actual_v_run, inl_lsb_run,
            marker="o", linestyle="None",
            ms=marker_size, lw=line_width, c=c_main, label="INL"
        )

    ax2.axhline(0, c=c_ideal, ls="--", lw=line_width)
    ax2.set_xlabel("Actual Input [V]")
    ax2.set_ylabel("INL [LSB]")
    ax2.grid(ls=":", lw=0.5, alpha=0.7)
    ax2.legend(fontsize="small")
    ax2.set_xlim(xlim)

    fig.suptitle(
        f"{runs}-Run DC Linearity ({steps} steps, ±{amplitude_vpp/2:.2f} V)\n",
        fontsize=12
    )

    if out_file:
        plt.savefig(out_file, dpi=300)
    if show:
        plt.show()
    plt.close(fig)
