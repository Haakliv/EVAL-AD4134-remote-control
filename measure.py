#!/usr/bin/env python3
"""
measure.py

Modular CLI framework for AD4134 EVM noise-floor test via ACE remote-control,
plus SFDR/THD/SINAD/ENOB dynamic performance test using Siglent SDG6022X.
Exports raw binary, computes noise-floor metrics, SFDR metrics, and optionally saves/plots histogram, FFT, and performance metrics.
"""
import argparse
import sys
import os
import time
import socket
import datetime

import numpy as np
import matplotlib.pyplot as plt

# Import the ACE Remote Control client for AD4134
sys.path.append(r"C:\Program Files (x86)\Analog Devices\ACE\Client")
import clr
clr.AddReference('AnalogDevices.Csa.Remoting.Clients')
import AnalogDevices.Csa.Remoting.Clients as adrc

# Siglent SDG6022X driver
from zolve_instruments import Sdg6022x

# UI and context paths for AD4134
UI_ROOT     = r"Root::System"
UI_BOARD    = UI_ROOT + r".Subsystem_1.AD4134 Eval Board"
UI_DRIVER   = UI_BOARD + r".AD4134"
UI_ANALYSIS = UI_DRIVER + r".AD4134 Analysis"

CTX_BOARD    = r"\System\Subsystem_1\AD4134 Eval Board"
CTX_DRIVER   = CTX_BOARD + r"\AD4134"
CTX_ANALYSIS = CTX_DRIVER + r"\AD4134 Analysis"

ACE_PLUGIN = 'AD4134 Eval Board'


def _measurement_folder():
    """Create timestamped Measurements folder and return its path."""
    base = os.path.dirname(os.path.abspath(__file__))
    meas_root = os.path.join(base, 'Measurements')
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = os.path.join(meas_root, ts)
    os.makedirs(folder, exist_ok=True)
    return folder

class ACEClient:
    """Thin wrapper around ACE Remote Control using Control API."""

    def __init__(self, host):
        mgr = adrc.ClientManager.Create()
        self.client = mgr.CreateRequestClient(host)
        # attach plugin
        self.client.NavigateToPath(UI_ROOT)
        self.client.AddHardwarePlugin(ACE_PLUGIN)
        # board reset
        self.client.set_ContextPath(CTX_BOARD)
        self.client.NavigateToPath(UI_BOARD)
        self.client.Run('@Reset()')
        time.sleep(2)
        # driver init
        self.client.set_ContextPath(CTX_DRIVER)
        self.client.NavigateToPath(UI_DRIVER)
        self.client.Run('@Initialization()')
        time.sleep(1)
        self.client.Run('@GetStatusFromBoard()')
        time.sleep(1)

    def get_local_ip(self):
        return socket.gethostbyname(socket.gethostname())

    def configure_board(self,
                        filter_type='Sinc6',
                        disable_channels='0,2,3',
                        odr=374000.0):
        """Set filter, data format, data frame, power-down channels, and output data rate."""
        fmap = {'Sinc1': '0', 'Sinc2': '1', 'Sinc6': '2'}
        code = fmap.get(filter_type, filter_type)
        self.client.set_ContextPath(CTX_DRIVER)
        self.client.NavigateToPath(UI_DRIVER)
        # power down channels
        for ch in disable_channels.split(','):
            self.client.Run(
                f'Evaluation.Control.SetIntParameter("PWRDN_CH{ch}",1,-1)'
            )
        # hardware filter
        self.client.Run(
            f'Evaluation.Control.SetIntParameter("virtual-parameter-filter-ch1",{code},-1)'
        )
        # parallel data format
        self.client.Run(
            'Evaluation.Control.SetIntParameter("virtual-parameter-data-format",2,-1)'
        )
        # 24-bit no CRC
        self.client.Run(
            'Evaluation.Control.SetIntParameter("virtual-parameter-data-frame",2,-1)'
        )
        # set output data rate
        self.client.Run(
            f'Evaluation.Control.SetDoubleParameter("virtual-parameter-output-data-rate",{odr},-1)'
        )
        # apply
        self.client.Run('@ApplySettings()')
        time.sleep(1)

    def capture(self, sample_count, timeout_ms=10000):
        """Capture raw binary to a .bin file and return its path."""
        folder = _measurement_folder()
        bin_path = os.path.join(folder, f"raw_{sample_count}.bin")
        # configure sample count
        self.client.set_ContextPath(CTX_DRIVER)
        self.client.NavigateToPath(UI_DRIVER)
        self.client.Run(
            f'Evaluation.Control.SetIntParameter("virtual-parameter-sample-count",{sample_count},-1)'
        )
        time.sleep(0.1)
        # raw capture
        self.client.AsyncRawCaptureToFile(bin_path, 'capture', 'false', 'true')
        self.client.WaitOnRawCaptureToFile(str(timeout_ms), 'capture', 'false', bin_path)
        return bin_path

# --- Data processing & plotting ---

def read_raw_samples(bin_file, big_endian, scale):
    data = np.fromfile(bin_file, dtype=np.uint8)
    samples = data.reshape(-1, 3)
    if big_endian:
        codes = (
            (samples[:,0].astype(np.int32) << 16) |
            (samples[:,1].astype(np.int32) <<  8) |
             samples[:,2].astype(np.int32)
        )
    else:
        codes = (
            (samples[:,2].astype(np.int32) << 16) |
            (samples[:,1].astype(np.int32) <<  8) |
             samples[:,0].astype(np.int32)
        )
    raw_counts = (codes << 8) >> 8
    return raw_counts.astype(float) * scale


def plot_raw(raw, out_file=None, show=False):
    plt.figure()
    plt.plot(raw)
    plt.xlabel('Sample Index'); plt.ylabel('Voltage [V]')
    plt.title('Noise Floor (Raw)')
    plt.tight_layout()
    if out_file:
        plt.savefig(out_file)
    if show:
        plt.show()


def plot_histogram(raw, bins, out_file=None, show=False):
    counts, edges = np.histogram(raw, bins=bins)
    hist_csv = None
    if out_file:
        hist_csv = out_file.replace('.png', '.csv')
        np.savetxt(hist_csv,
                   np.vstack((edges[:-1], counts)).T,
                   delimiter=',', header='bin_start,count', comments='')
    plt.figure()
    plt.hist(raw, bins=bins)
    plt.xlabel('Voltage [V]'); plt.ylabel('Count')
    plt.title('Histogram')
    plt.tight_layout()
    if out_file:
        plt.savefig(out_file)
    if show:
        plt.show()
    return hist_csv


def plot_fft(raw, fs, out_file=None, show=False):
    N = raw.size
    freqs = np.fft.rfftfreq(N, d=1.0/fs)
    fft_vals = np.abs(np.fft.rfft(raw))
    fft_csv = None
    if out_file:
        fft_csv = out_file.replace('.png', '.csv')
        np.savetxt(fft_csv,
                   np.vstack((freqs, fft_vals)).T,
                   delimiter=',', header='frequency,value', comments='')
    plt.figure()
    plt.plot(freqs, fft_vals)
    plt.xlabel('Frequency [Hz]'); plt.ylabel('Magnitude')
    plt.title('FFT')
    plt.tight_layout()
    if out_file:
        plt.savefig(out_file)
    if show:
        plt.show()
    return fft_csv


def compute_metrics(freqs, spectrum, f0, num_harmonics=5):
    idx_f = np.argmin(np.abs(freqs - f0))
    P1 = spectrum[idx_f]**2
    spectrum_no_dc = spectrum.copy()
    spectrum_no_dc[0] = 0
    # noise+distortion
    P_total = np.sum(spectrum_no_dc**2)
    P_nd = P_total - P1
    sinad = 10 * np.log10(P1 / P_nd)
    enob = (sinad - 1.76) / 6.02
    # THD
    P_h = 0.0
    for n in range(2, num_harmonics+1):
        P_h += spectrum[np.argmin(np.abs(freqs - n*f0))]**2
    thd = 10 * np.log10(P_h / P1)
    # SFDR
    spurs = spectrum_no_dc.copy()
    spurs[idx_f] = 0
    max_spur = np.max(spurs)
    sfdr = 20 * np.log10(spectrum[idx_f] / max_spur)
    return sfdr, thd, sinad, enob

# ---- CLI setup and handlers ----

def setup_parsers():
    parser = argparse.ArgumentParser(description='measure.py CLI')
    subs = parser.add_subparsers(dest='cmd', required=True)
    # noise-floor
    nf = subs.add_parser('noise-floor', help='Noise floor test')
    nf.add_argument('-n','--samples', type=int, default=131072)
    nf.add_argument('--ace-host', type=str, default='localhost:2357')
    nf.add_argument('--scale', type=float, default=4.8828125e-07)
    nf.add_argument('--big-endian', action='store_true')
    nf.add_argument('--histogram', action='store_true')
    nf.add_argument('--hist-bins', type=int, default=100)
    nf.add_argument('--fft', action='store_true')
    nf.add_argument('--plot', action='store_true')
    nf.add_argument('--show', action='store_true')
    # sfdr
    sf = subs.add_parser('sfdr', help='SFDR test')
    sf.add_argument('--sdg-host', type=str, default='192.168.1.100')
    sf.add_argument('--channel', type=int, choices=[1,2], default=1)
    sf.add_argument('--freq', type=float, default=1000.0)
    sf.add_argument('--amplitude', type=float, default=1.0)
    sf.add_argument('--offset', type=float, default=0.0)
    sf.add_argument('--ace-host', type=str, default='localhost:2357')
    sf.add_argument('-n','--samples', type=int, default=131072)
    sf.add_argument('--scale', type=float, default=4.8828125e-07)
    sf.add_argument('--big-endian', action='store_true')
    sf.add_argument('--plot', action='store_true')
    sf.add_argument('--show', action='store_true')
    sf.add_argument('--no-board', action='store_true')
    return parser


def main():
    parser = setup_parsers()
    args = parser.parse_args()

    if args.cmd == 'noise-floor':
        ace = ACEClient(args.ace_host)
        print(f"ACE @ {args.ace_host}, IP {ace.get_local_ip()}")
        ace.configure_board()
        bin_file = ace.capture(args.samples)
        print(f"Binary saved: {bin_file}")

        raw = read_raw_samples(bin_file, args.big_endian, args.scale)
        mean, std, ptp = raw.mean(), raw.std(), np.ptp(raw)
        print("\nResults:")
        print(f"  mean_V: {mean:.6e} V")
        print(f"  std_V : {std:.6e} V")
        print(f"  pp_V  : {ptp:.6e} V")

        if args.plot or args.show:
            png = bin_file.replace('.bin', '_raw.png') if args.plot else None
            plot_raw(raw, out_file=png, show=args.show)
            if png: print(f"Raw plot saved: {png}")
            elif args.show: print("Raw plot displayed")

        if args.histogram:
            h_png = bin_file.replace('.bin', '_hist.png') if args.plot else None
            h_csv = plot_histogram(raw, args.hist_bins, out_file=h_png, show=args.show)
            if h_png: print(f"Histogram plot saved: {h_png}")
            if h_csv: print(f"Histogram data saved: {h_csv}")
            elif args.show: print("Histogram displayed")

        if args.fft:
            f_png = bin_file.replace('.bin', '_fft.png') if args.plot else None
            f_csv = plot_fft(raw, out_file=f_png, show=args.show)
            if f_png: print(f"FFT plot saved: {f_png}")
            if f_csv: print(f"FFT data saved: {f_csv}")
            elif args.show: print("FFT displayed")

    elif args.cmd == 'sfdr':
        sdg = Sdg6022x(args.sdg_host)
        sdg.set_waveform('SINE', args.channel)
        sdg.set_frequency(args.freq, args.channel)
        sdg.set_amplitude(args.amplitude, args.channel)
        sdg.set_offset(args.offset, args.channel)
        sdg.enable_output(args.channel)
        print(f"SDG6022X @ {args.sdg_host}, ch{args.channel}: {args.freq}Hz, {args.amplitude}Vpp offset {args.offset}V")

        if args.no_board:
            f_act = sdg.get_frequency(args.channel)
            a_act = sdg.get_amplitude(args.channel)
            o_act = sdg.get_offset(args.channel)
            print(f"Verified generator: {f_act:.3f}Hz, {a_act:.3f}Vpp, {o_act:.3f}V")
            sdg.disable_output(args.channel)
            return

        ace = ACEClient(args.ace_host)
        print(f"ACE @ {args.ace_host}, IP {ace.get_local_ip()}")
        ace.configure_board(odr=args.odr)
        bin_file = ace.capture(args.samples)
        print(f"Binary saved: {bin_file}")

        raw = read_raw_samples(bin_file, args.big_endian, args.scale)
        freqs, spec, _ = plot_fft(raw, fs=args.odr, show=False)
        sfdr_v, thd_v, sinad_v, enob_v = compute_metrics(freqs, spec, args.freq)
        print("\nSFDR Test Results:")
        print(f"  SFDR  : {sfdr_v:.2f} dB")
        print(f"  THD   : {thd_v:.2f} dB")
        print(f"  SINAD : {sinad_v:.2f} dB")
        print(f"  ENOB  : {enob_v:.2f} bits")

        if args.plot or args.show:
            perf_png = bin_file.replace('.bin', '_perf.png')
            plt.figure(); plt.plot(freqs, spec)
            plt.xlabel('Frequency [Hz]'); plt.ylabel('Magnitude')
            plt.title(f'SFDR: {sfdr_v:.2f} dB, THD: {thd_v:.2f} dB')
            plt.tight_layout()
            if args.plot: plt.savefig(perf_png); print(f"Performance plot saved: {perf_png}")
            if args.show: plt.show()

        sdg.disable_output(args.channel)

if __name__ == '__main__':
    main()
