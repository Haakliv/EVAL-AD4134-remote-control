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
from scipy.fft import rfft, rfftfreq
from scipy.optimize import nnls
from multimeter import Dmm6500Controller


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
    compute_settling_time,
    compute_mean_settling_time,
    find_spur_rms,
)
from plotting    import (
    plot_agg_fft,
    plot_agg_histogram,
    plot_settling_time,
    plot_freq_response,
    plot_dc_linearity_summary,
    plot_fft_with_metrics,
)

# --- Constants & defaults ----------------------------------------------------
ACE_HOST_DEFAULT  = 'localhost:2357'
SDG_HOST_DEFAULT  = '172.16.1.56'
B29_HOST_DEFAULT  = '169.254.5.2'
DMM_IP_DEFAULT    = '169.254.15.212'

ADC_LSB           = MAX_INPUT_RANGE / (2**(ADC_RES_BITS - 1))
DEFAULT_STEP_VPP  = MAX_INPUT_RANGE * 0.9
LOW_PCT = 50.0

ADC_ODR_CODE_DEFAULT    = 1      # 1.25 MHz
ADC_FILTER_CODE_DEFAULT = 2      # Sinc6

SAMPLES_DEFAULT      = 131072
HIST_BINS_DEFAULT    = 1000
SETTLING_THRESH_FRAC = 0.01
SWEEP_POINTS_DEFAULT = 50

SWITCHING_FREQ = 295400   # Hz, power-supply switching frequency for spur detection
FILTER_BW_FACTORS = 232630 # Hz  # Sinc6 BW factor (−3 dB) at 1.25MHz ODR for spur integration

DEFAULT_INL_STEPS = 4096          # 2 mV step over 8.192 V span
DEFAULT_RUNS  = 3

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
            ace, args.samples, ADC_LSB, output_dir=os.getcwd()
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
            "Power-supply spur RMS @ %.1f kHz: %.3f uVrms",
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


_DBV  = lambda v: 20*np.log10(v)

# -------------------------------------------------------------------------
#  Dynamic-performance test (external generator, no AWG control)
# -------------------------------------------------------------------------
def run_sfdr(args, logger, ace):
    """
    Measures SFDR / THD / SINAD / ENOB with a manually controlled sine source.
    The --runs flag averages multiple captures.

    Required CLI flags:
        --freq        target test frequency (Hz)
        --amplitude   expected differential Vpp at the ADC inputs
        --offset      common-mode DC offset (informational only)
        --runs        number of repeat captures
        --plot / --show  optional FFT plot
    """

    # ---------- ADC parameters --------------------------------------------
    fs   = SLAVE_ODR_MAP[args.odr_code]         # output data rate (Hz)
    n    = args.samples

    # ---------- Coherent bin calculation ----------------------------------
    k_bin = int(round(args.freq * n / fs))
    f_coh = k_bin * fs / n                      # actual coherent frequency

    # ---------- Inline 7-term Blackman–Harris window ----------------------
    a = [0.2712203606, 0.4334446123, 0.21800412,
         0.0657853433, 0.0107618673, 0.0007700125,
         0.0000136809]
    k_vec = np.arange(n)
    win   = sum(a[m] * np.cos(2.0 * np.pi * m * (k_vec - n / 2) / n)
                for m in range(7))
    wc    = win.sum() / n                       # amplitude-loss factor
    MAIN  = 15                                  # +/- main-lobe bins

    # ---------- –3 dB bandwidth lookup (Sinc-6) ---------------------------
    pb_hz = 69793

    # ---------- Logging ---------------------------------------------------
    logger.info("Dynamic performance test: tone %.0f Hz (coherent %.6f Hz), %.2f Vpp-diff, "
                "%d runs, ODR %.0f Hz, %d samples",
                args.freq, f_coh, args.amplitude,
                args.runs, fs, n)

    if args.no_board:
        logger.warning("--no-board specified; skipping ADC capture")
        return None, None, None, None

    ace.setup_capture(n, args.odr_code)

    # ---------- Capture loop ----------------------------------------------
    spectra = []
    for run_idx in range(1, args.runs + 1):
        raw = capture_samples(ace_client=ace,
                              sample_count=n,
                              output_dir=os.getcwd())          # volts
        spec = rfft(raw * win)
        mag  = 2.0 * np.abs(spec) / (n * wc) / np.sqrt(2.0)    # Vrms / bin
        spectra.append(mag)
        logger.debug("Run %d fundamental: %.2f dBV",
                     run_idx, 20.0 * np.log10(mag[k_bin]))

    mag_avg = np.mean(spectra, axis=0)
    mag_avg = np.maximum(mag_avg, 1e-20)                       # avoid log(0)

    # ---------- Metrics ----------------------------------------------------
    def calc_metrics(freq_axis, mag_vec, k_fund, passband_hz,
                     H=5, dc_bins=10):
        mask = np.ones_like(mag_vec, dtype=bool)
        mask[:dc_bins] = False                                 # DC guard
        mask[max(0, k_fund - MAIN):k_fund + MAIN + 1] = False  # fundamental
        for h in range(2, H + 1):                              # H2..H5
            bin_h = h * k_fund
            if bin_h < len(mask):
                mask[max(0, bin_h - MAIN):bin_h + MAIN + 1] = False
        mask &= freq_axis <= passband_hz                       # pass-band

        P1   = (mag_vec[k_fund - MAIN:k_fund + MAIN + 1] ** 2).sum()
        fund = np.sqrt(P1)
        spur = mag_vec[mask].max()
        sfdr = 20.0 * np.log10(fund / spur)

        Ph = sum((mag_vec[h * k_fund - MAIN:
                          h * k_fund + MAIN + 1] ** 2).sum()
                 for h in range(2, H + 1) if h * k_fund < len(mag_vec))
        thd   = 10.0 * np.log10(Ph / P1) if Ph > 0.0 else -np.inf

        Pnd   = (mag_vec[mask] ** 2).sum()
        sinad = 10.0 * np.log10(P1 / Pnd)
        enob  = (sinad - 1.76) / 6.02
        return sfdr, thd, sinad, enob

    freqs = rfftfreq(n, 1.0 / fs)
    sfdr, thd, sinad, enob = calc_metrics(freqs, mag_avg, k_bin, pb_hz)

    logger.info("SFDR %.2f dB, THD %.2f dB, SINAD %.2f dB, ENOB %.2f bits",
                sfdr, thd, sinad, enob)

    # ---------- Optional FFT plot -----------------------------------------
    if args.plot or args.show:
        filt_name = SINC_FILTER_MAP[args.filter_code]
        plot_fft_with_metrics(
            freqs, mag_avg, fs,
            sfdr, thd, sinad, enob,
            runs=args.runs,
            tone_freq=f_coh,
            amplitude_vpp=args.amplitude,
            filt=filt_name,
            out_file="sfdr_fft.png",
            show=args.show,
            xlim=(0.0, pb_hz / 1e3),        # limits in kHz
        )

    return sfdr, thd, sinad, enob


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
                ace, wave_gen, f, odr_hz,
                args.amplitude, args.offset,
                args.settle_cycles, args.capture_cycles,
                logger
            )
            power_runs[run_idx, i] = vrms**2    # Store power

    wave_gen.disable(1)
    wave_gen.disable(2)

    mean_power = power_runs.mean(axis=0)

    vrms_avg = np.sqrt(mean_power)
    input_peak = args.amplitude
    gains      = vrms_avg * np.sqrt(2) / input_peak

    ref = np.median(gains[:max(3, args.points // 20)])  # Median of first ~5 %
    gains_norm = gains / ref
    gdB_norm   = 20 * np.log10(gains_norm)

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

def run_dc_tests(args, logger, ace):
    """
    Repeated DC sweep – two use-cases, selected by --no-board:

      (1) Normal INL test (default)
          • SMU / DMM are the references
          • ADC board is the DUT

      (2) Source-linearity test (--no-board)
          • Commanded sweep is the reference
          • SMU (and, if present, DMM) are the DUTs
    """
    start_time       = time.time()
    next_time_update = start_time + 300                      # 5-min heartbeat

    # ---------------- Instrument Setup ----------------
    smu = B2912A(args.resource)
    dmm = None if args.no_dmm else Dmm6500Controller(args.dmm_ip)

    dmm_fixed_range = 100
    centre          = float(getattr(args, "offset", 0.0))    # volts
    raw_amp         = float(args.amplitude)

    # Ensure the sweep never exceeds the fixed DMM range +/-dmm_fixed_range.
    max_pos_headroom = dmm_fixed_range - centre
    max_neg_headroom = dmm_fixed_range + centre
    safe_amp         = 2 * min(max_pos_headroom, max_neg_headroom)
    if safe_amp <= 0:
        raise ValueError(f"Offset {centre} V is outside the +/-{dmm_fixed_range} V "
                         "meter range.")
    amplitude = min(raw_amp, safe_amp)
    v_start, v_stop = centre - amplitude / 2, centre + amplitude / 2

    if dmm:
        dmm.configure_for_precise_dc(nplc=5, autozero=True,
                                     dmm_range=dmm_fixed_range)
        logger.info("DMM: %d V range, 5 PLC, AZER=ON, rear terminals",
                    dmm_fixed_range)
    else:
        logger.info("--no-dmm – using SMU sensed voltage only")

    logger.info("Sweep: %.3f Vpp centred on %.3f V  "
                "(start %.3f V to stop %.3f V)",
                amplitude, centre, v_start, v_stop)

    smu.output_on()

    # ---------------- Sweep Grid ----------------
    steps = args.steps or DEFAULT_INL_STEPS
    runs  = args.runs  or DEFAULT_RUNS
    sweep_voltages = np.linspace(v_stop, v_start, steps)     # descending sweep

    settle_delay_s = 0.05                                    # 50 ms guard

    if not args.no_board:
        ace.setup_capture(args.samples, args.odr_code)

    run_stats = {"dmm": [], "smu": []}

    try:
        for run_idx in range(1, runs + 1):
            logger.info("=== Run %d / %d ===", run_idx, runs)

            actual_v_smu, actual_v_dmm, adc_v = [], [], []

            for k, v in enumerate(sweep_voltages, 1):
                # ---------- 1 – command SMU & wait deterministically ----------
                smu.smu.write("*OPC")                         # arm completion flag
                smu.set_voltage(v, current_limit=0.01)
                smu.smu.query("*OPC?")                        # wait for settle
                while True:
                    oper = int(smu.smu.query("STAT:OPER:COND?"))
                    if oper & 0b1 == 0:                       # bit 0 = Operation-Complete
                        break
                    time.sleep(0.002)

                time.sleep(settle_delay_s)                    # extra 50 ms

                # ---------- 2 – SMU self-measurement ----------
                actual_v_smu.append(smu.measure_voltage())

                # ---------- 3 – DMM reference ----------
                if dmm:
                    _ = dmm.measure_voltage_dc()              # throw-away
                    mean_v, _, _ = dmm.measure_voltage_avg(
                        n_avg=10, delay=0.05)
                    actual_v_dmm.append(mean_v)

                # ---------- 4 – ADC board (if enabled) ----------
                if not args.no_board:
                    raw = capture_samples(ace, args.samples, ADC_LSB,
                                          output_dir=os.getcwd())
                    adc_v.append(np.mean(raw))
                else:
                    # In source-linearity mode we use the *commanded* voltage
                    # as the reference X-axis, store it in adc_v for simplicity.
                    adc_v.append(v)

                # ---------- 5 – progress heartbeat ----------
                now = time.time()
                if now >= next_time_update or k == steps:
                    logger.info("Step %d/%d  (%.4f V)", k, steps, v)
                    next_time_update = now + 300

                print(f"Step {k}/{steps}: Commanded {v:+.6f} V",
                      end="\r", flush=True)

            # ---------- per-run analysis ----------
            if not adc_v:
                logger.warning("No measurement data collected; skipping analysis")
                continue

            adc_arr = np.asarray(adc_v)                       # common X-axis

            if args.no_board:
                # --- 4-wire SMU and external DMM versus commanded sweep ---
                def analyse_src(label, meas_v):
                    meas_arr = np.asarray(meas_v)
                    gain, offset = np.polyfit(adc_arr, meas_arr, 1)
                    resid        = meas_arr - (gain * adc_arr + offset)
                    inl_lsb      = resid / ADC_LSB
                    inl_ppm      = resid / (MAX_INPUT_RANGE * 2) * 1e6
                    typ_inl_ppm  = np.mean(np.abs(inl_ppm))
                    rs = dict(label       = label,
                              gain        = gain,
                              offset_uV   = offset * 1e6,
                              max_inl_ppm = np.max(np.abs(inl_ppm)),
                              typ_inl_ppm = typ_inl_ppm,
                              rms_inl_lsb = np.sqrt(np.mean(inl_lsb**2)))
                    logger.info(
                        "Run %d [%s]  Gain=%.6f  Offset=%.1f uV  "
                        "Max|INL|=%.2f ppm  Typ|INL|=%.2f ppm",
                        run_idx, label, gain, rs['offset_uV'],
                        rs['max_inl_ppm'], rs['typ_inl_ppm']
                    )
                    plot_dc_linearity_summary(
                        actual_v_run = adc_arr,               # commanded
                        adc_v_run    = meas_arr,              # measured
                        fit_line_run = gain * adc_arr + offset,
                        inl_lsb_run  = inl_lsb,
                        avg_gain         = gain,
                        avg_offset_uV    = offset * 1e6,
                        avg_max_inl_ppm  = rs['max_inl_ppm'],
                        avg_rms_inl_lsb  = rs['rms_inl_lsb'],
                        runs             = 1,
                        amplitude_vpp    = amplitude,
                        steps            = steps,
                        out_file         = f"dc_plot_src_{label.lower()}.png",
                        show             = getattr(args, "show_plots", False),
                    )
                    return rs

                run_stats["smu"].append(analyse_src("SMU", actual_v_smu))
                if dmm:
                    run_stats["dmm"].append(analyse_src("DMM", actual_v_dmm))

            else:
                # --- Normal ADC INL measurement path ---
                def analyse(label, ref_v):
                    v_arr = np.asarray(ref_v)
                    gain, offset = np.polyfit(v_arr, adc_arr, 1)
                    resid        = adc_arr - (gain * v_arr + offset)
                    inl_lsb      = resid / ADC_LSB
                    inl_ppm      = resid / (MAX_INPUT_RANGE * 2) * 1e6
                    typ_inl_ppm  = np.mean(np.abs(inl_ppm))
                    rs = dict(label       = label,
                              gain        = gain,
                              offset_uV   = offset * 1e6,
                              max_inl_ppm = np.max(np.abs(inl_ppm)),
                              typ_inl_ppm = typ_inl_ppm,
                              rms_inl_lsb = np.sqrt(np.mean(inl_lsb**2)))
                    logger.info(
                        "Run %d [%s]  Gain=%.6f  Offset=%.1f uV  "
                        "Max|INL|=%.2f ppm  Typ|INL|=%.2f ppm",
                        run_idx, label, gain, rs['offset_uV'],
                        rs['max_inl_ppm'], rs['typ_inl_ppm']
                    )
                    return rs

                run_stats["smu"].append(analyse("SMU", actual_v_smu))
                if dmm:
                    run_stats["dmm"].append(analyse("DMM", actual_v_dmm))

    finally:
        smu.output_off()
        smu.close()

    # ---------- aggregate & summary (print once per tag) ----------
    printed_summary = set()

    def agg(tag):
        if tag in printed_summary or not run_stats[tag]:
            return
        printed_summary.add(tag)

        g  = np.mean([r["gain"]        for r in run_stats[tag]])
        ou = np.mean([r["offset_uV"]   for r in run_stats[tag]])
        ip = np.mean([r["max_inl_ppm"] for r in run_stats[tag]])
        ir = np.mean([r["rms_inl_lsb"] for r in run_stats[tag]])
        tp = np.mean([r["typ_inl_ppm"] for r in run_stats[tag]])

        logger.info("*** %s reference (avg over %d runs) ***",
                    tag.upper(), len(run_stats[tag]))
        logger.info("Gain err    : %.3f %%", (g - 1) * 100)
        logger.info("Offset      : %.1f uV", ou)
        logger.info("Max |INL|   : %.2f ppm FS", ip)
        logger.info("Typical |INL| : %.2f ppm FS", tp)
        logger.info("RMS  INL    : %.3f LSB", ir)

    agg('dmm')
    agg('smu')

# =============================================================================
# Argument-parser construction
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
    parser.add_argument('--adc-channel', type=int, choices=[0, 1, 2, 3], default=1,
                        help='ADC channel to use (0-3)')

def add_common_plot_args(parser):
    """Plotting/display flags shared by all sub-commands."""
    parser.add_argument('--plot', action='store_true', help='save plots to files')
    parser.add_argument('--show', action='store_true', help='display plots on screen')

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

    # --- SFDR / THD / ENOB ---------------------------------------------------
    sf = subs.add_parser('sfdr', help='Dynamic-performance test')
    sf.add_argument('--no-board', dest='no_board', action='store_true',
                    help='skip ADC capture')
    sf.add_argument('--freq', type=float, required=True,
                    help='test frequency [Hz]')
    sf.add_argument('--amplitude', type=float, default=DEFAULT_STEP_VPP,
                    help='expected differential Vpp at the ADC inputs')

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

    # --- DC tests ----------------------------------------------------
    dct = subs.add_parser('dc-test', help='DC measurement test (DMM + Source)')
    dct.add_argument('--no-board', dest='no_board', action='store_true',
                    help='Dry-run: skip ADC capture')
    dct.add_argument('--dmm-ip', type=str, default=DMM_IP_DEFAULT,
                     help='DMM IP address (default: %s)' % DMM_IP_DEFAULT)
    dct.add_argument('--resource', type=str,
                     default=f'TCPIP0::{B29_HOST_DEFAULT}::inst0::INSTR',
                     help='B2912A VISA resource')
    dct.add_argument('--amplitude', type=float,
        default=MAX_INPUT_RANGE * 2,
        help='Total sweep amplitude [V]. Sweep is from -amplitude/2 to +amplitude/2.')
    dct.add_argument('--steps', type=int,   default=4096,
                 help='number of voltage points (default≈2 mV step)')
    dct.add_argument('--no-dmm', dest='no_dmm', action='store_true',
                    help='skip DMM measurements (use ADC only)')
    dct.add_argument('--offset', type=float, default=0.0,
                    help='Common-mode DC offset [V] (default: 0.0 V)')
    add_common_adc_args(dct)
    add_common_plot_args(dct)
    dct.set_defaults(runs=1, samples=4096)

    return p

# =============================================================================
# Main
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
        # List all channels as integers
        all_channels = [0, 1, 2, 3]
        # Exclude the selected channel
        disable_list = [str(ch) for ch in all_channels if ch != args.adc_channel]
        # Join them into a string
        disable_channels = ','.join(disable_list)
        ace.configure_board(
            filter_code=args.filter_code,
            disable_channels=disable_channels
        )


    # --- Dispatch ------------------------------------------------------------
    if   args.cmd == 'noise-floor':   run_noise_floor(args, logger, ace)
    elif args.cmd == 'sfdr':          run_sfdr(args, logger, ace)
    elif args.cmd == 'settling-time': run_settling_time(args, logger, ace)
    elif args.cmd == 'freq-response': run_freq_response(args, logger, ace)
    elif args.cmd == 'dc-test':       run_dc_tests(args, logger, ace)
    else:
        logger.error("Unknown command (use -h for help)")

# -----------------------------------------------------------------------------
if __name__ == '__main__':
    main()
