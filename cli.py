#!/usr/bin/env python3
import argparse
import numpy as np
import os
import datetime
import logging

from scipy.signal import welch

from ace_client import (
    ACEClient,
    MAX_INPUT_RANGE,
    SLAVE_ODR_MAP,
    SINC_FILTER_MAP,
    ADC_RES_BITS,
)
from generator import WaveformGenerator
from b2912a_source import B2912A
from acquisition import capture_samples
from processing import (
    fft_spectrum,
    compute_metrics,
    compute_settling_time,
    compute_bandwidth,
    compute_dc_gain_offset,
)
from plotting import (
    plot_agg_fft,
    plot_agg_histogram,
    plot_settling,
    plot_freq_response,
    plot_dc_gain,
)

# -- Constants and defaults ----------------------------------------------------------
ACE_HOST_DEFAULT     = 'localhost:2357'
SDG_HOST_DEFAULT     = '172.16.1.56'
B29_HOST_DEFAULT     = '169.254.5.2'
ADC_LSB              = MAX_INPUT_RANGE / (2 ** (ADC_RES_BITS - 1))

# Default codes
ADC_ODR_CODE_DEFAULT    = 1   # 1.25 MHz
ADC_FILTER_CODE_DEFAULT = 2   # Sinc6

SAMPLES_DEFAULT      = 131_072
HIST_BINS_DEFAULT    = 100
SETTLING_THRESH_FRAC = 0.01
SWEEP_POINTS_DEFAULT = 50


# Adds ADC-related arguments to the parser
def add_common_adc_args(parser):
    parser.add_argument(
        '--ace-host', dest='ace_host', type=str,
        default=ACE_HOST_DEFAULT,
        help='ACE server address'
    )
    parser.add_argument(
        '--odr-code', type=int, choices=list(SLAVE_ODR_MAP.keys()),
        default=ADC_ODR_CODE_DEFAULT,
        help=('ADC ODR code (0–13): ' +
              ', '.join(f"{c}={r:.0f}Hz" for c, r in SLAVE_ODR_MAP.items()))
    )
    parser.add_argument(
        '--filter-code', type=int, choices=list(SINC_FILTER_MAP.keys()),
        default=ADC_FILTER_CODE_DEFAULT,
        help=('ADC filter code (0–4): ' +
              ', '.join(f"{c}={name}" for c, name in SINC_FILTER_MAP.items()))
    )
    parser.add_argument(
        '-n', '--samples', type=int,
        default=SAMPLES_DEFAULT,
        help='number of samples to capture'
    )
    parser.add_argument(
        '--runs', type=int, default=5,
        help='number of repeat measurements to perform'
    )


# Adds shared plotting flags to the parser
def add_common_plot_args(parser):
    parser.add_argument('--plot', action='store_true', help='save plots to files')
    parser.add_argument('--show', action='store_true', help='display plots on screen')


def setup_parsers():
    parser = argparse.ArgumentParser(description='CLI')
    subs = parser.add_subparsers(dest='cmd', required=True)

    # -- Noise floor ----------------------------------------------------------
    nf = subs.add_parser('noise-floor', help='Noise floor test')
    add_common_adc_args(nf)
    add_common_plot_args(nf)
    nf.add_argument('--histogram', action='store_true', help='plot histogram')
    nf.add_argument('--hist-bins', type=int, default=HIST_BINS_DEFAULT,
                    help='number of bins for histogram')
    nf.add_argument('--fft', action='store_true', help='perform FFT')

    # -- SFDR, THD, ENOB ---------------------------------------------------------------
    sf = subs.add_parser('sfdr', help='SFDR/SINAD/ENOB test')
    sf.add_argument('--sdg-host', dest='sdg_host', type=str,
                    default=SDG_HOST_DEFAULT,
                    help='SDG server address'
    )
    sf.add_argument('--channel', type=int, choices=[1, 2], default=1,
                    help='SDG channel'
    )
    sf.add_argument('--freq', type=float, default=1000.0,
                    help='sine frequency in Hz'
    )
    sf.add_argument('--amplitude', type=float, default=1.0,
                    help='Vpp of sine input'
    )
    sf.add_argument('--offset', type=float, default=0.0,
                    help='DC offset in V'
    )
    sf.add_argument('--no-board', dest='no_board', action='store_true',
                    help='skip board measurement verification'
    )
    add_common_adc_args(sf)
    add_common_plot_args(sf)

    # -- Settling time -------------------------------------------------------
    st = subs.add_parser('settling-time', help='Transient settling time test')
    st.add_argument('--sdg-host', dest='sdg-host', type=str,
                    default=SDG_HOST_DEFAULT,
                    help='SDG server address'
    )
    st.add_argument('--channel', type=int, choices=[1, 2], default=1,
                    help='SDG channel'
    )
    st.add_argument('--amplitude', type=float, default=8.0,
                    help='Voltage pp of step (V)'
    )
    st.add_argument('--offset', type=float, default=0.0,
                    help='Offset voltage (V)'
    )
    st.add_argument('--frequency', type=float, default=1.0,
                    help='Step repetition rate in Hz'
    )
    st.add_argument('--threshold', type=float,
                    default=SETTLING_THRESH_FRAC,
                    help='Settling threshold fraction of half-step'
    )
    add_common_adc_args(st)
    add_common_plot_args(st)

    # -- Frequency response ------------------------------------------------------
    fr = subs.add_parser('freq-response', help='Frequency response and AC gain test')
    fr.add_argument('--sdg-host', dest='sdg-host', type=str,
                    default=SDG_HOST_DEFAULT,
                    help='SDG server address'
    )
    fr.add_argument('--channel', type=int, choices=[1, 2], default=1,
                    help='SDG channel'
    )
    fr.add_argument('--freq-start', dest='freq_start', type=float, default=10.0,
                    help='start frequency in Hz'
    )
    fr.add_argument('--freq-stop', dest='freq_stop', type=float, default=10000.0,
                    help='stop frequency in Hz'
    )
    fr.add_argument('--points', type=int,
                    default=SWEEP_POINTS_DEFAULT,
                    help='number of sweep points'
    )
    fr.add_argument('--amplitude', type=float, default=1.0,
                    help='Vpp of sine input'
    )
    fr.add_argument('--offset', type=float, default=0.0,
                    help='DC offset in V'
    )
    add_common_adc_args(fr)
    add_common_plot_args(fr)

    # -- DC-gain/offset ----------------------------------------------------------
    dcg = subs.add_parser(
        'dc-gain', help='Measure DC gain and offset with ADC'
    )
    dcg.add_argument(
        'voltages', nargs='+', type=float,
        help='List of DC voltages to apply via SMU'
    )
    dcg.add_argument(
        '--resource', type=str, default=f'TCPIP0::{B29_HOST_DEFAULT}::inst0::INSTR',
        help='TCPIP VISA resource of B2912A (e.g. TCPIP0::192.168.1.100::inst0::INSTR)'
    )
    dcg.add_argument(
        '-no-board', dest='no_board', action='store_true',
        help='Skip ADC capture; only apply voltages'
    )
    # ADC capture arguments
    add_common_adc_args(dcg)
    add_common_plot_args(dcg)

    return parser


def main():
    args = setup_parsers().parse_args()

    # Create a per-test output directory and switch into it
    measurements_root = 'Measurements'
    os.makedirs(measurements_root, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(measurements_root, f"{args.cmd}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    # switch into run directory so all files land here
    os.chdir(run_dir)

    # --- Logging setup ----------------------------------------------------
    log_file = f"{args.cmd}_results.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()

    logger.info(f"Starting '{args.cmd}' test @ {datetime.datetime.now().isoformat()}")
    # translate codes into actual values
    odr_rate = SLAVE_ODR_MAP[args.odr_code]
    filter_type = SINC_FILTER_MAP[args.filter_code]

    ace = ACEClient(args.ace_host)
    ace.configure_board(
        filter_code=args.filter_code,
        disable_channels='0,2,3'
    )

    # Prepare kwargs for the other commands
    common_capture_kwargs = dict(
        ace_client  = ace,
        sample_count= args.samples,
        scale       = ADC_LSB,
        odr_code    = args.odr_code,
    )

    # ------------------- Noise floor -------------------
    if args.cmd == 'noise-floor':
        odr_rate = SLAVE_ODR_MAP[args.odr_code]
        filter_name = SINC_FILTER_MAP[args.filter_code]
        
        logger.info(f"Noise Floor Test on EVAL-AD4134: runs={args.runs}, ODR={odr_rate:.0f}Hz, filter={filter_type}")
        stds = []
        raw_runs = []
        welch_sum = None

        for i in range(1, args.runs + 1):
            raw = capture_samples(
                ace,
                args.samples,
                ADC_LSB,
                args.odr_code,
                output_dir=os.getcwd()
            )
            raw_runs.append(raw)
            mean = np.mean(raw)
            std = np.std(raw, ddof=0)
            ptp = np.ptp(raw)
            stds.append(std)

            freqs, Pxx = welch(raw, fs=odr_rate,
                                nperseg=len(raw)//4,
                                noverlap=len(raw)//8)
            welch_sum = Pxx if welch_sum is None else welch_sum + Pxx

            nsd = np.sqrt(Pxx)
            med_nsd = np.median(nsd[1:-1])
            logger.info(f"Run {i}: mean={mean:.3e} V, std={std:.3e} V, ptp={ptp:.3e} V, NSD_med={med_nsd:.3e} V/sqrt(Hz)")

        mean_std = np.mean(stds)
        spread = np.std(stds, ddof=1)
        logger.info(f"RMS noise: {mean_std:.3e} V ± {spread:.3e} V")

        Pxx_avg = welch_sum / args.runs
        nsd_avg = np.sqrt(Pxx_avg)
        med_nsd_avg = np.median(nsd_avg[1:-1])
        logger.info(f"Median NSD: {med_nsd_avg:.3e} V/sqrt(Hz)")

        if args.plot or args.show:
            if args.histogram:
                plot_agg_histogram(
                    np.concatenate(raw_runs), args.hist_bins,
                    args.runs, odr_rate, filter_name,
                    out_file='agg_hist.png', show=args.show
                )
            if args.fft:
                mag = np.sqrt(Pxx_avg * odr_rate / 2)
                plot_agg_fft(
                    freqs, mag, args.runs,
                    odr_rate, filter_name,
                    out_file='agg_fft.png', show=args.show
                )
        return

    # ------------------- SFDR, THD, ENOB -------------------
    elif args.cmd == 'sfdr':
        print('=== Dynamic Performance Test ===')
        print('Connecting to function generator...')
        gen = WaveformGenerator(args.sdg_host)
        gen.sine(args.channel, args.freq, args.amplitude, args.offset)
        print(f"SDG Output: {args.freq:.3f} Hz, {args.amplitude:.3f} Vpp, "
              f"offset {args.offset:.3f} V")
        if args.no_board:
            actual_f = gen.sdg.get_frequency(args.channel)
            actual_a = gen.sdg.get_amplitude(args.channel)
            actual_o = gen.sdg.get_offset(args.channel)
            print(f"Verified: {actual_f:.3f} Hz, {actual_a:.3f} Vpp, "
                  f"offset {actual_o:.3f} V")
            gen.disable(args.channel)
            return
        raw = capture_samples(**common_capture_kwargs)
        freqs, spectrum = fft_spectrum(raw, odr_rate)
        sfdr_v, thd_v, sinad_v, enob_v = compute_metrics(freqs, spectrum, args.freq)
        print(f"SFDR={sfdr_v:.2f} dB, THD={thd_v:.2f} dB, "
              f"SINAD={sinad_v:.2f} dB, ENOB={enob_v:.2f} bits")
        gen.disable(args.channel)

    # ------------------- Settling time -------------------
    elif args.cmd == 'settling-time':
        print('=== Settling Time Test ===')
        gen = WaveformGenerator(args.sdg_host)
        gen.pulse(args.channel, args.frequency, args.amplitude, args.offset)
        print(f"SDG Step: {args.frequency:.3f} Hz, {args.amplitude:.3f} Vpp, "
              f"offset {args.offset:.0f} V")
        raw = capture_samples(**common_capture_kwargs)
        settling = compute_settling_time(
            raw, odr_rate, args.threshold * (args.amplitude/2)
        )
        if settling is not None:
            print(f"Settling time: {settling*1e3:.2f} ms")
        else:
            print("Settling time not detected within capture window")
        if args.plot or args.show:
            plot_settling(raw, odr_rate,
                          out_file = args.plot and f"settling_{args.samples}.png",
                          show     = args.show)
        gen.disable(args.channel)

    # ------------------- Frequency response -------------------
    elif args.cmd == 'freq-response':
        print('=== Frequency Response Test ===')
        freqs = np.linspace(args.freq_start, args.freq_stop, args.points)
        gains = []
        gen = WaveformGenerator(args.sdg_host)
        for f in freqs:
            gen.sine(args.channel, f, args.amplitude, args.offset)
            raw = capture_samples(**common_capture_kwargs)
            f_bins, spectrum = fft_spectrum(raw, odr_rate)
            idx = np.argmin(np.abs(f_bins - f))
            gain = (2.0 * spectrum[idx] / raw.size) / (args.amplitude / 2.0)
            gains.append(gain)
            gen.disable(args.channel)
        cutoff = compute_bandwidth(freqs, gains)
        print(f"-3 dB bandwidth: {cutoff:.2f} Hz @ {odr_rate:.0f}Hz ODR, "
              f"filter={filter_type}")
        if args.plot or args.show:
            plot_freq_response(freqs, gains,
                               out_file = args.plot and f"freq_resp_{args.points}.png",
                               show     = args.show)
            
    # ------------------- DC gain/offset -------------------
    if args.cmd == 'dc-gain':
        print('=== DC Gain and Offset Test (EVAL-AD4134) ===')
        print(f"SMU resource: {args.resource}")
        print(f"DMM resource: {args.dmm_resource}")

        voltages = args.voltages
        num_pts = len(voltages)
        print(f"Applying {num_pts} DC points from {voltages[0]:.6f} V to {voltages[-1]:.6f} V")

        smu = B2912A(args.resource)

        adc_means = []
        actual_vs = []
        trigger_delay = 0.2

        for v in voltages:
            smu.sweep_voltage(
                start=v,
                stop=v,
                points=1,
                current_limit=0.01,
                range_mode='AUTO',
                trigger_count=1,
                trigger_delay=trigger_delay,
                trigger_period=None
            )
            smu.smu.query('*OPC?')

            if not args.no_board:
                raw = capture_samples(
                    ace_host=args.ace_host,
                    sample_count=args.samples,
                    scale=ADC_LSB,
                    odr_code=args.odr_code,
                    filter_code=args.filter_code,
                )
                mean_v = np.mean(raw)
                adc_means.append(mean_v)
                print(f"ADC Mean: {mean_v:.6f} V")

        smu.close()

        if args.no_board:
            return

        results = compute_dc_gain_offset(actual_vs, None, adc_means)
        print(f"\nGain: {results['gain']:.6f}, Offset: {results['offset']:.6f}, R²: {results['r2']:.4f}")

        if args.plot or args.show:
            out_file = args.plot and f"dc_gain_{num_pts}.png"
            plot_dc_gain(actual_vs, adc_means, out_file=out_file, show=args.show)

        return

    print("Unknown command. Use -h for help.")

if __name__ == '__main__':
    main()
