#!/usr/bin/env python3
import argparse
import numpy as np
import os
import time

from ace_client import (
    ACEClient,
    MAX_INPUT_RANGE,
    SLAVE_ODR_MAP,
    SINC_FILTER_MAP,
    ADC_RES_BITS,
)
from generator import WaveformGenerator
from b2912a_source import B2912A
from acquisition import capture_samples, read_raw_samples, measure_dmm
from processing import (
    compute_noise_floor_metrics,
    fft_spectrum,
    compute_metrics,
    compute_settling_time,
    compute_bandwidth,
    compute_dc_gain_offset,
)
from plotting import (
    plot_raw,
    plot_histogram,
    plot_fft,
    plot_settling,
    plot_freq_response,
)

# === Constants and Defaults ===
ACE_HOST_DEFAULT     = 'localhost:2357'
SDG_HOST_DEFAULT     = '172.16.1.56'
ADC_LSB              = MAX_INPUT_RANGE / (2 ** (ADC_RES_BITS - 1))

# Default codes
ADC_ODR_CODE_DEFAULT    = 7   # 363.636 kHz
ADC_FILTER_CODE_DEFAULT = 2   # Sinc6

SAMPLES_DEFAULT      = 131_072
HIST_BINS_DEFAULT    = 100
SETTLING_THRESH_FRAC = 0.01
SWEEP_POINTS_DEFAULT = 50

# === Agilent Defaults ===
DEFAULT_SMU_RESOURCE = 'TCPIP0::169.254.5.2::inst0::INSTR'

def add_common_adc_args(parser):
    """Adds ADC-related arguments to the parser."""
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


def add_common_plot_args(parser):
    """Adds shared plotting flags to the parser."""
    parser.add_argument('--plot', action='store_true', help='save plots to files')
    parser.add_argument('--show', action='store_true', help='display plots on screen')


def setup_parsers():
    parser = argparse.ArgumentParser(description='CLI')
    subs = parser.add_subparsers(dest='cmd', required=True)

    # -- noise-floor ----------------------------------------------------------
    nf = subs.add_parser('noise-floor', help='Noise floor test')
    add_common_adc_args(nf)
    add_common_plot_args(nf)
    nf.add_argument('--histogram', action='store_true', help='plot histogram')
    nf.add_argument('--hist-bins', type=int, default=HIST_BINS_DEFAULT,
                    help='number of bins for histogram')
    nf.add_argument('--fft', action='store_true', help='perform FFT')

    # -- sfdr ---------------------------------------------------------------
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

    # -- settling-time -------------------------------------------------------
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

    # -- freq-response ------------------------------------------------------
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

    # -- dc-gain ----------------------------------------------------------
    # DC gain/offset test
    dcg = subs.add_parser(
        'dc-gain', help='Measure DC gain and offset with ADC'
    )
    dcg.add_argument(
        'voltages', nargs='+', type=float,
        help='List of DC voltages to apply via SMU'
    )
    dcg.add_argument(
        '--resource', type=str, default=DEFAULT_SMU_RESOURCE,
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

    # translate codes into actual values
    odr_rate = SLAVE_ODR_MAP[args.odr_code]
    filter_type = SINC_FILTER_MAP[args.filter_code]

    # Prepare kwargs for the other commands
    common_capture_kwargs = dict(
        ace_host    = args.ace_host,
        sample_count= args.samples,
        scale       = ADC_LSB,
        odr_code    = args.odr_code,
        filter_code = args.filter_code,
    )

    # ------------------- noise-floor -------------------
    if args.cmd == 'noise-floor':
        ace = ACEClient(args.ace_host)
        print(f"ACE @ {args.ace_host}, IP {ace.get_local_ip()}")
        print(f"Using ODR={odr_rate:.0f}Hz (code={args.odr_code}), "
              f"filter={filter_type} (code={args.filter_code})")

        # Configure and capture with the single ACEClient
        ace.configure_board(
            filter_code      = args.filter_code,
            disable_channels = '0,2,3',
            odr_code         = args.odr_code
        )
        bin_path = ace.capture(
            sample_count = args.samples,
            odr_code     = args.odr_code
        )
        print(f"Binary saved: {bin_path}")

        os.chdir(os.path.dirname(bin_path))

        # Read raw 24-bit samples into volts
        raw = read_raw_samples(
            bin_file   = bin_path,
            scale      = ADC_LSB
        )

        # Compute and print metrics
        mean, std, ptp = compute_noise_floor_metrics(raw)
        print(f"Noise floor: mean={mean:.3e} V, std={std:.3e} V, pp={ptp:.3e} V")

        # Optional plots
        if args.plot or args.show:
            plot_raw(raw,
                     out_file = args.plot and f"raw_{args.samples}.png",
                     show     = args.show)
        if args.histogram:
            plot_histogram(raw,
                           bins     = args.hist_bins,
                           out_file = args.plot and f"hist_{args.samples}.png",
                           show     = args.show)
        if args.fft:
            freqs, spectrum = fft_spectrum(raw, odr_rate)
            plot_fft(freqs, spectrum,
                     out_file = args.plot and f"fft_{args.samples}.png",
                     show     = args.show)

    # ------------------- sfdr -------------------
    elif args.cmd == 'sfdr':
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

    # ------------------- settling-time -------------------
    elif args.cmd == 'settling-time':
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

    # ------------------- freq-response -------------------
    elif args.cmd == 'freq-response':
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
            
    # ------------------- dc-gain -------------------
    elif args.cmd == 'dc-gain':
        print('=== DC Gain and Offset Test ===')
        print(f"SMU LAN resource: {args.resource}")
        print(f"Voltages: {args.voltages}")
        if args.no_board:
            print('--no-board: skipping ADC capture')

        # Build capture params
        capture_kwargs = dict(
            ace_host=args.ace_host,
            sample_count=args.samples,
            scale=ADC_LSB,
            odr_code=args.odr_code,
            filter_code=args.filter_code,
        )

        # Initialize SMU over LAN
        smu = B2912A(args.resource)
        applied = []
        adc_readings = []

        for V in args.voltages:
            print(f"\nApplying {V:.6f} V...")
            smu.set_voltage(V, current_limit=0.01, range_mode='AUTO')
            smu.output_on()
            time.sleep(0.2)

            if not args.no_board:
                raw = capture_samples(**capture_kwargs)
                v_meas = np.mean(raw)
                print(f"ADC mean: {v_meas:.6f} V")
                if args.plot:
                    from plotting import plot_histogram
                    fname = f"hist_{V:.6f}.png"
                    plot_histogram(raw,
                                   bins=args.hist_bins,
                                   out_file=fname,
                                   show=args.show)
                    print(f"Histogram saved to {fname}")
            else:
                v_meas = None

            smu.output_off()
            applied.append(V)
            adc_readings.append(v_meas)

        smu.close()

        # Summary
        print('\n--- Summary ---')
        print('Applied(V)   ADC(V)')
        for a, m in zip(applied, adc_readings):
            print(f"{a:10.6f}   {m or 'N/A':>8}")

        if args.no_board:
             print("\n--no-board: skipping gain/offset calculation")
             return

        # Compute gain/offset
        results = compute_dc_gain_offset(applied, None, adc_readings)
        print(f"\nGain: {results['gain']:.6f}, Offset: {results['offset']:.6f}, R2: {results['r2']:.4f}")
        return

    print("Unknown command. Use -h for help.")

if __name__ == '__main__':
    main()
