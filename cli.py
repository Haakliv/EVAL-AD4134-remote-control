#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# EVAL-AD4134 Measurement CLI
# -----------------------------------------------------------------------------
import argparse
import datetime
import logging
import os
import numpy as np
import time
from scipy.signal import welch
import math

# --- Project-local modules ---------------------------------------------------
from ace_client import (
    ACEClient,
    MAX_INPUT_RANGE,
    SLAVE_ODR_MAP,
    SINC_FILTER_MAP,
    ADC_RES_BITS,
)
from generator   import WaveformGenerator
from b2912a_source import B2912A
from acquisition import capture_samples
from processing  import (
    fft_spectrum,
    compute_metrics,
    compute_settling_time,
    compute_mean_settling_time,
    compute_dc_gain_offset,
    find_spur_rms,
)
from plotting    import (
    plot_agg_fft,
    plot_agg_histogram,
    plot_settling_time,
    plot_freq_response,
    plot_dc_gain,
)

# --- Constants & defaults ----------------------------------------------------
ACE_HOST_DEFAULT  = 'localhost:2357'
SDG_HOST_DEFAULT  = '172.16.1.56'
B29_HOST_DEFAULT  = '169.254.5.2'

ADC_LSB           = MAX_INPUT_RANGE / (2**(ADC_RES_BITS - 1))
DEFAULT_STEP_VPP  = MAX_INPUT_RANGE * 0.9
LOW_PCT = 50.0

ADC_ODR_CODE_DEFAULT    = 1      # 1.25 MHz
ADC_FILTER_CODE_DEFAULT = 2      # Sinc6

SAMPLES_DEFAULT      = 131_072
HIST_BINS_DEFAULT    = 1000
SETTLING_THRESH_FRAC = 0.01
SWEEP_POINTS_DEFAULT = 50

SWITCHING_FREQ = 295400   # Hz, power-supply switching frequency for spur detection
FILTER_BW_FACTORS = 232630 # Hz  # Sinc6 BW factor (−3 dB) at 1.25MHz ODR for spur integration

# =============================================================================
# Helper utilities
# =============================================================================
def add_common_adc_args(parser):
    """ADC-related flags shared by all sub-commands."""
    parser.add_argument('--ace-host', dest='ace_host', type=str,
                        default=ACE_HOST_DEFAULT, help='ACE server address')
    parser.add_argument('--odr-code', type=int, choices=list(SLAVE_ODR_MAP.keys()),
                        default=ADC_ODR_CODE_DEFAULT,
                        help=('ADC ODR code (0-13): ' +
                              ', '.join(f"{c}={r:.0f} Hz"
                                        for c, r in SLAVE_ODR_MAP.items())))
    parser.add_argument('--filter-code', type=int,
                        choices=list(SINC_FILTER_MAP.keys()),
                        default=ADC_FILTER_CODE_DEFAULT,
                        help=('ADC filter code (0-4): ' +
                              ', '.join(f"{c}={name}"
                                        for c, name in SINC_FILTER_MAP.items())))
    parser.add_argument('-n', '--samples', type=int, default=SAMPLES_DEFAULT,
                        help='number of samples to capture')
    parser.add_argument('--runs', type=int, default=5,
                        help='number of repeat measurements')

def add_common_plot_args(parser):
    """Plotting/display flags shared by all sub-commands."""
    parser.add_argument('--plot', action='store_true', help='save plots to files')
    parser.add_argument('--show', action='store_true', help='display plots on screen')

# -----------------------------------------------------------------------------
def load_awg_cal(cal_path, target_freqs, odr_rate):
    """Load AWG baseline PSD and interpolate onto `target_freqs`."""
    cal = np.load(cal_path, allow_pickle=True)
    P_awg   = cal['Pxx']
    f_awg   = cal['freqs']
    meta    = cal['meta'].item() if 'meta' in cal.files else {}
    if meta.get('fs', odr_rate) != odr_rate:
        logging.warning("AWG-cal fs mismatch (cal %.0f Hz, test %.0f Hz)",
                        meta.get('fs'), odr_rate)
    # Convert PSD → magnitude spectrum
    mag_awg = np.sqrt(P_awg * odr_rate / 2.0)
    return np.interp(target_freqs, f_awg, mag_awg)

# =============================================================================
# Measurement routines
# =============================================================================
# -- Noise floor --------------------------------------------------------------
def run_noise_floor(args, logger, ace):
    """Aggregate-statistics noise-floor measurement."""
    ace.setup_capture(args.samples, args.odr_code)
    odr     = SLAVE_ODR_MAP[args.odr_code]
    filt    = SINC_FILTER_MAP[args.filter_code]
    logger.info("Noise-floor: runs=%d, ODR=%.0f Hz, filter=%s",
                args.runs, odr, filt)

    welch_sum, stds, raw_runs = None, [], []
    for i in range(1, args.runs + 1):
        raw = capture_samples(
            ace, args.samples, ADC_LSB, args.odr_code, output_dir=os.getcwd()
        )
        raw_runs.append(raw)
        mean = np.mean(raw)
        std  = np.std(raw, ddof=0)
        ptp  = np.ptp(raw)
        stds.append(std)

        f, Pxx = welch(raw, fs=odr,
                       nperseg=len(raw)//4, noverlap=len(raw)//8)
        welch_sum = Pxx if welch_sum is None else welch_sum + Pxx
        nsd_med = np.median(np.sqrt(Pxx)[1:-1])
        logger.info("Run %d: mean=%.3e V, std=%.3e V, ptp=%.3e V, "
                    "NSD_med=%.3e V/sqrt(Hz)", i, mean, std, ptp, nsd_med)

    rms  = np.mean(stds)
    spread = np.std(stds, ddof=1)
    logger.info("RMS noise: %.3e V ± %.3e V", rms, spread)

    Pxx_avg  = welch_sum / args.runs
    nsd_avg  = np.sqrt(Pxx_avg)
    mag_avg = np.sqrt(Pxx_avg * odr / 2.0)
    med_nsd  = np.median(nsd_avg[1:-1])
    logger.info("Median NSD: %.3e V/sqrt(Hz)", med_nsd)

    try:
        spur_rms_v, spur_freq = find_spur_rms(f, mag_avg, SWITCHING_FREQ, span_hz=10e3)
        spur_uvrms = spur_rms_v * 1e6
        logger.info(
            "Power-supply spur RMS @ %.1f kHz: %.3f µVrms",
            spur_freq/1e3, spur_uvrms
        )
    except ValueError as e:
        logger.warning("Spur detection failed: %s", e)

    if args.plot or args.show:
        if args.histogram:
            plot_agg_histogram(
                np.concatenate(raw_runs), args.hist_bins,
                args.runs, odr, filt,
                out_file='agg_hist.png', show=args.show
            )
        if args.fft:
            mag = np.sqrt(Pxx_avg * odr / 2)
            plot_agg_fft(
                f, mag, args.runs, odr, filt,
                out_file='agg_fft.png', show=args.show
            )

# -- AWG baseline PSD capture -------------------------------------------------
def run_gen_spectrum(args, logger, ace):
    """Capture AWG-only PSD and store in NPZ for later subtraction."""
    odr = SLAVE_ODR_MAP[args.odr_code]
    gen = WaveformGenerator(args.sdg_host)
    fs_vpp = MAX_INPUT_RANGE * 10 ** (-0.5 / 20)
    logger.info("Using %.6f Vpp (-0.5 dBFS) instead of %.6f Vpp", fs_vpp, args.amplitude)
    gen.sine(args.channel, args.freq, fs_vpp, args.offset)
    logger.info("Capturing AWG baseline PSD: f=%.2f Hz, %.2f Vpp, offset=%.2f V",
                args.freq, args.amplitude, args.offset)

    acc = None
    for _ in range(args.runs):
        raw = capture_samples(
            ace, args.samples, ADC_LSB, args.odr_code, output_dir=os.getcwd()
        )
        f, Pxx = welch(raw, fs=odr,
                       nperseg=len(raw)//4, noverlap=len(raw)//8)
        acc = Pxx if acc is None else acc + Pxx
    Pxx_avg = acc / args.runs
    np.savez('awg_psd.npz', Pxx=Pxx_avg, freqs=f,
             meta=dict(fs=odr, f0=args.freq,
                       amp=args.amplitude, off=args.offset))
    logger.info("Saved AWG baseline → awg_psd.npz")
    gen.disable(args.channel)

# -- SFDR / THD / ENOB --------------------------------------------------------
def run_sfdr(args, logger, ace):
    """Dynamic-performance test with optional AWG baseline subtraction."""
    odr  = SLAVE_ODR_MAP[args.odr_code]
    filt = SINC_FILTER_MAP[args.filter_code]
    logger.info("SFDR: f=%.2f Hz, %.2f Vpp, offset=%.2f V, ODR=%.0f kHz, filter=%s",
                args.freq, args.amplitude, args.offset, odr, filt)

    gen = WaveformGenerator(args.sdg_host)
    gen.sine(args.channel, args.freq, args.amplitude, args.offset)

    if args.no_board:
        logger.info("Generator verification only; skipping ADC capture.")
        logger.info("Actual AWG settings: f=%.3f Hz, Vpp=%.3f V, "
                    "offset=%.3f V",
                    gen.sdg.get_frequency(args.channel),
                    gen.sdg.get_amplitude(args.channel),
                    gen.sdg.get_offset(args.channel))
        gen.disable(args.channel)
        return

    raw = capture_samples(
        ace, args.samples, ADC_LSB, args.odr_code, output_dir=os.getcwd()
    )
    freqs, spectrum = fft_spectrum(raw, odr)

    # --- AWG baseline subtraction (optional) --------------------------------
    if args.awg_cal:
        mag_awg = load_awg_cal(args.awg_cal, freqs, odr)
        spectrum = np.clip(spectrum - mag_awg, 0, None)
        logger.info("Applied AWG baseline subtraction (%s)", args.awg_cal)

    sfdr_v, thd_v, sinad_v, enob_v = compute_metrics(freqs, spectrum, args.freq)
    logger.info("SFDR=%.2f dB, THD=%.2f dB, SINAD=%.2f dB, ENOB=%.2f bits",
                sfdr_v, thd_v, sinad_v, enob_v)
    gen.disable(args.channel)

# -- Settling‑time ------------------------------------------------------------
def run_settling_time(args, logger, ace):
    """
    AD4134 settling-time test using simple sample-based edge/flattop detection.
    """
    odr        = SLAVE_ODR_MAP[args.odr_code]
    filt       = SINC_FILTER_MAP[args.filter_code]
    Ts_us      = 1e6 / odr

    # +-1 LSB_eff band  (approx 60uV @1.25MSPS, 17.3 ENOB)
    LSB_eff_uV = 2 * MAX_INPUT_RANGE / 2**17.3

    logger.info("Settling test: ODR=%.0fHz, %s, runs=%d, Vpp=%.2f, freq=%.1fHz",
                odr, filt, args.runs, args.amplitude, args.frequency)

    ace.setup_capture(args.samples, args.odr_code)

    gen = WaveformGenerator(args.sdg_host, "PULSE", args.offset)
    gen.pulse_diff(args.frequency, args.amplitude,
                   low_pct=50.0, edge_time=2e-9)
    time.sleep(0.5)

    raw_runs     = []
    start_idxs   = []
    end_idxs     = []
    start_times  = []
    end_times    = []
    times_ns     = []

    for run in range(1, args.runs + 1):
        raw = capture_samples(
            ace, args.samples, ADC_LSB,
            output_dir=os.getcwd()
        )
        raw_runs.append(raw)

        idx_start, idx_end, Ts = compute_settling_time(
            raw, fs=odr, tol_uV=LSB_eff_uV
        )

        if idx_start is None or idx_end is None:
            logger.warning("Run %d: settling not detected", run)
            continue

        # record per-run
        t_start = idx_start * Ts
        t_end   = idx_end   * Ts
        dt_us   = t_end - t_start

        start_idxs.append(idx_start)
        end_idxs.append(idx_end)
        start_times.append(t_start)
        end_times.append(t_end)
        times_ns.append(dt_us * 1e3)

        logger.info("Run %d: Delta=%.2fus", run, dt_us)

    gen.disable(1)
    gen.disable(2)

    if not start_idxs or not end_idxs:
        logger.error("No valid measurements obtained")
        return

    # compute and log the mean settling across all runs
    mean_delta, time_vec, mean_seg = compute_mean_settling_time(raw_runs, start_idxs, end_idxs, Ts_us, pad=2)
    arr    = np.array(times_ns)
    mean_ns = arr.mean()
    std_ns  = arr.std(ddof=1) if arr.size > 1 else 0.0
    ci95_ns = 1.96 * std_ns / np.sqrt(arr.size) if arr.size > 1 else 0.0

    # log mean pm95% CI and std in us
    logger.info(
        "Mean settling: %.2f us +- %.2f us (95%% CI), std=%.2f us over %d runs",
        mean_ns*1e-3, ci95_ns*1e-3, std_ns*1e-3, arr.size
    )

    if (args.plot or args.show) and raw_runs and start_idxs:
        plot_file = os.path.join(os.getcwd(), 'settling.png')
        plot_settling_time(
            raw_runs, start_idxs, end_idxs,
            Ts_us, time_vec, mean_seg, mean_delta,
            filt, odr,
            args.frequency, args.amplitude, args.runs,
            out_file=plot_file,
            show=args.show
        )

def measure_tone(
        args,
        ace_client,
        wave_gen: "WaveformGenerator",
        freq_hz: float,
        odr_hz: float,
        amplitude: float,
        offset: float,
        settle_cycles: int,
        capture_cycles: int,
        logger,
        ch_pos: int = 1,
        ch_neg: int = 2,
):
    wave_gen.sine_diff(freq_hz, amplitude, offset,
                       ch_pos=ch_pos, ch_neg=ch_neg)

    time.sleep(settle_cycles / freq_hz)

    samples = int(odr_hz * capture_cycles / freq_hz)
    logger.info(f"Capture @ {freq_hz:.1f} Hz = {samples} samples")

    if samples > SAMPLES_DEFAULT:
        # Shrink capture_cycles so it fits in one shot
        new_capture_cycles = int(SAMPLES_DEFAULT * freq_hz / odr_hz)
        logger.warning(
            "tone %.1f Hz: %d samples would exceed buffer; "
            "reducing capture_cycles from %d to %d",
            freq_hz, samples, capture_cycles, new_capture_cycles
        )
        capture_cycles = new_capture_cycles
        samples = SAMPLES_DEFAULT

    raw = capture_samples(
        ace_client=ace_client,
        sample_count=samples,
        output_dir=os.getcwd()
    )

    vrms = float(np.sqrt(np.mean(raw.astype(np.float64) ** 2)))

    logger.debug("tone %.1f Hz, Vrms %.6f, %d samples",
                 freq_hz, vrms, raw.size)
    return freq_hz, vrms

# TODO: Fikse konsistent amplitude
# -- Frequency response ------------------------------------------------------
def run_freq_response(args, logger, ace):
    odr_hz      = SLAVE_ODR_MAP[args.odr_code]
    filter_name = SINC_FILTER_MAP[args.filter_code]

    logger.info(
        "Step-sine response: %.1f Hz -> %.1f Hz  "
        "(%d pts, %d runs)  ODR=%.0f kHz, Filter=%s, Vpp=%.2f",
        args.freq_start, args.freq_stop,
        args.points, args.runs,
        odr_hz/1e3, filter_name, args.amplitude*2
    )

    ace.setup_capture(args.samples, args.odr_code)

    freqs = np.logspace(
        math.log10(args.freq_start),
        math.log10(args.freq_stop),
        args.points
    )

    wave_gen = WaveformGenerator(args.sdg_host)
    wave_gen.enable(1)
    wave_gen.enable(2)

    power_runs = np.zeros((args.runs, args.points), dtype=np.float64)

    for run_idx in range(args.runs):
        logger.info("Run %d / %d", run_idx + 1, args.runs)
        for i, f in enumerate(freqs):
            _, vrms = measure_tone(
                args, ace, wave_gen, f, odr_hz,
                args.amplitude, args.offset,
                args.settle_cycles, args.capture_cycles,
                logger
            )
            power_runs[run_idx, i] = vrms**2    # Store power

    wave_gen.disable(1)
    wave_gen.disable(2)

    mean_power = power_runs.mean(axis=0)

    if args.runs > 1:
        std_power = power_runs.std(axis=0, ddof=1)      # Unbiased std
        vrms_err  = 0.5 * std_power / np.sqrt(mean_power)   # Standard error
    else:
        std_power = np.zeros_like(mean_power)
        vrms_err  = np.zeros_like(mean_power)           # No uncertainty

    vrms_avg = np.sqrt(mean_power)
    input_peak = args.amplitude / 2.0
    gains      = vrms_avg * np.sqrt(2) / input_peak
    gain_err   = (vrms_err / gains) if args.runs > 1 else None

    # TODO: Fikse at man kan kun ha 1 run

    ref = np.median(gains[:max(3, args.points // 20)])  # Median of first ~5 %
    gains_norm = gains / ref
    gdB_norm   = 20 * np.log10(gains_norm)
    err_dB     = 20 * np.log10(1 + gain_err)            # Small-angle approx

    idx = np.where(gdB_norm <= -3.0)[0]
    if idx.size:
        k   = idx[0]
        f3dB = np.interp(-3.0,
                         [gdB_norm[k-1], gdB_norm[k]],
                         [freqs[k-1],    freqs[k]])
        logger.info(
            "-3 dB bandwidth: %.0f Hz  (mean of %d runs, 95%% CI shown below)",
            f3dB, args.runs
        )
    else:
        logger.info("-3 dB point not in sweep range")

    passband_dB = gdB_norm[ freqs < 1000 ]              # First decade
    mu   = passband_dB.mean()
    s    = passband_dB.std(ddof=1)
    ci95 = 1.96 * s / np.sqrt(len(passband_dB))
    logger.info(
        "Pass-band gain: %.2f dB +- %.2f dB (95%% CI), "
        "std = %.2f dB over %d runs",
        mu, ci95, s, args.runs
    )

    plot_freq_response(freqs,
                   gains_norm,
                   runs=args.runs,
                   amplitude_vpp=args.amplitude,
                   out_file=(args.plot if isinstance(args.plot, str)
                             else "step_bode.png"),
                   show=args.show)

# -- DC gain / offset ---------------------------------------------------------
def run_dc_gain(args, logger, ace):
    """DC gain/offset linear-fit measurement."""
    logger.info("DC gain/offset: SMU=%s", args.resource)
    voltages = args.voltages
    logger.info("Applying %d DC points (%.3f→%.3f V)",
                len(voltages), voltages[0], voltages[-1])

    smu = B2912A(args.resource)
    adc_means, actual_vs = [], []

    for v in voltages:
        smu.sweep_voltage(
            start=v, stop=v, points=1, current_limit=0.01,
            range_mode='AUTO', trigger_count=1,
            trigger_delay=0.2, trigger_period=None
        )
        smu.smu.query('*OPC?')

        if args.no_board:
            actual_vs.append(v)
            continue

        raw = capture_samples(
            ace, args.samples, ADC_LSB, args.odr_code, output_dir=os.getcwd()
        )
        mean_v = np.mean(raw)
        actual_vs.append(v)
        adc_means.append(mean_v)
        logger.info("V_set=%.6f V → ADC_mean=%.6f V", v, mean_v)

    smu.close()

    if args.no_board:
        return

    res = compute_dc_gain_offset(actual_vs, None, adc_means)
    logger.info("Gain=%.6f, Offset=%.6f, R²=%.4f",
                res['gain'], res['offset'], res['r2'])

    if args.plot or args.show:
        plot_dc_gain(
            actual_vs, adc_means,
            out_file=args.plot and f"dc_gain_{len(voltages)}.png",
            show=args.show
        )

# =============================================================================
# Argument-parser construction
# =============================================================================
def setup_parsers():
    p = argparse.ArgumentParser(description='EVAL-AD4134 test CLI')
    subs = p.add_subparsers(dest='cmd', required=True)

    # --- Noise floor ---------------------------------------------------------
    nf = subs.add_parser('noise-floor', help='Noise-floor test')
    add_common_adc_args(nf)
    add_common_plot_args(nf)
    nf.add_argument('--histogram', action='store_true', help='plot histogram')
    nf.add_argument('--hist-bins', type=int, default=HIST_BINS_DEFAULT,
                    help='bins for histogram')
    nf.add_argument('--fft', action='store_true', help='perform FFT')

    # --- AWG baseline capture -----------------------------------------------
    gs = subs.add_parser('gen-spectrum', help='Capture AWG baseline PSD')
    gs.add_argument('--sdg-host', dest='sdg_host', type=str,
                    default=SDG_HOST_DEFAULT, help='SDG address')
    gs.add_argument('--channel', type=int, choices=[1, 2], default=1,
                    help='SDG channel')
    gs.add_argument('--freq', type=float, default=1_000.0, help='sine freq [Hz]')
    gs.add_argument('--amplitude', type=float, default=1.0, help='Vpp')
    gs.add_argument('--offset', type=float, default=0.0, help='DC offset [V]')
    add_common_adc_args(gs)

    # --- SFDR / THD / ENOB ---------------------------------------------------
    sf = subs.add_parser('sfdr', help='Dynamic-performance test')
    sf.add_argument('--sdg-host', dest='sdg_host', type=str,
                    default=SDG_HOST_DEFAULT, help='SDG address')
    sf.add_argument('--channel', type=int, choices=[1, 2], default=1,
                    help='SDG channel')
    sf.add_argument('--freq', type=float, default=1_000.0, help='sine freq [Hz]')
    sf.add_argument('--amplitude', type=float, default=1.0, help='Vpp')
    sf.add_argument('--offset', type=float, default=0.0, help='DC offset [V]')
    sf.add_argument('--no-board', dest='no_board', action='store_true',
                    help='skip ADC capture')
    sf.add_argument('--awg-cal', type=str,
                    help='NPZ file with AWG baseline PSD (from gen-spectrum)')
    add_common_adc_args(sf)
    sf.set_defaults(odr_code=7)
    add_common_plot_args(sf)

    # --- Settling-time -------------------------------------------------------
    st = subs.add_parser('settling-time', help='Transient settling-time test')
    st.add_argument('--sdg-host', dest='sdg_host', type=str,
                    default=SDG_HOST_DEFAULT, help='SDG address')
    st.add_argument('--channel', type=int, choices=[1, 2], default=1,
                    help='SDG channel')
    st.add_argument('--amplitude', type=float, default=DEFAULT_STEP_VPP, help='step Vpp')
    st.add_argument('--offset', type=float, default=0.0, help='offset [V]')
    st.add_argument('--frequency', type=float, default=100.0, help='rep rate [Hz]')
    st.add_argument('--no-board', dest='no_board', action='store_true',
                    help='skip ADC capture (AWG-only verification)')
    st.add_argument('--start-wait', type=float, default=0.5,
                help='seconds to wait after enabling the AWG before the first capture')
    add_common_adc_args(st)
    add_common_plot_args(st)

    # --- Frequency-response --------------------------------------------------
    fr = subs.add_parser('freq-response', help='Continuous-chirp freq-response gain sweep')
    fr.add_argument('--sdg-host', dest='sdg_host', type=str,
                    default=SDG_HOST_DEFAULT, help='SDG address')
    fr.add_argument('--channel', type=int, choices=[1, 2], default=1,
                    help='AWG channel')
    fr.add_argument('--freq-start', dest='freq_start', type=float, default=400.0,
                    help='Sweep start frequency [Hz]')
    fr.add_argument('--freq-stop', dest='freq_stop', type=float, default=625000.0,
                    help='Sweep stop frequency [Hz]')
    fr.add_argument('--points', dest='points', type=int, default=250,
                    help='Amount of step points to sweep')
    fr.add_argument('--amplitude', dest='amplitude', type=float, default=1.0,
                    help='Differential Vpp of the sine sweep')
    fr.add_argument('--offset', dest='offset', type=float, default=0.0,
                    help='Common-mode DC offset [V]')
    fr.add_argument('--settle_cycles', dest='settle_cycles', type=int, default=8,
                    help='Number of cycles to wait before capturing')
    fr.add_argument('--capture_cycles', dest='capture_cycles', type=int, default=32,
                    help='Number of cycles to capture')
    fr.add_argument('--no-board', dest='no_board', action='store_true',
                    help='Dry-run: skip ADC capture')
    add_common_adc_args(fr)
    add_common_plot_args(fr)

    # --- DC gain / offset ----------------------------------------------------
    dcg = subs.add_parser('dc-gain', help='DC gain & offset test')
    dcg.add_argument('voltages', nargs='+', type=float,
                     help='list of DC voltages [V]')
    dcg.add_argument('--resource', type=str,
                     default=f'TCPIP0::{B29_HOST_DEFAULT}::inst0::INSTR',
                     help='B2912A VISA resource')
    dcg.add_argument('-no-board', dest='no_board', action='store_true',
                     help='skip ADC capture')
    add_common_adc_args(dcg)
    add_common_plot_args(dcg)

    return p

# =============================================================================
# Main entry-point
# =============================================================================
def main():
    args = setup_parsers().parse_args()

    # --- Per‑run output directory -------------------------------------------
    root = 'Measurements'
    os.makedirs(root, exist_ok=True)
    tstamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(root, f"{args.cmd}_{tstamp}")
    os.makedirs(run_dir, exist_ok=True)
    os.chdir(run_dir)

    # --- Logging -------------------------------------------------------------
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=[logging.FileHandler(f"{args.cmd}_results.log"),
                  logging.StreamHandler()]
    )
    logger = logging.getLogger()
    logger.info("Starting '%s' test", args.cmd)

    # --- ADC board configuration --------------------------------------------
    ace = None                     # only connect if we actually need the board
    if not getattr(args, 'no_board', False):
        ace = ACEClient(args.ace_host)
        ace.configure_board(filter_code=args.filter_code,
                            disable_channels='0,2,3')

    # --- Dispatch ------------------------------------------------------------
    if   args.cmd == 'noise-floor':   run_noise_floor(args, logger, ace)
    elif args.cmd == 'gen-spectrum':  run_gen_spectrum(args, logger, ace)
    elif args.cmd == 'sfdr':          run_sfdr(args, logger, ace)
    elif args.cmd == 'settling-time': run_settling_time(args, logger, ace)
    elif args.cmd == 'freq-response': run_freq_response(args, logger, ace)
    elif args.cmd == 'dc-gain':       run_dc_gain(args, logger, ace)
    else:
        logger.error("Unknown command (use -h for help)")

# -----------------------------------------------------------------------------
if __name__ == '__main__':
    main()
