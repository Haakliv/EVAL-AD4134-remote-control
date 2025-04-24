#!/usr/bin/env python3
"""
measure.py

Modular CLI framework for AD4134 EVM noise-floor test via ACE remote-control.
Exports raw binary, computes noise-floor metrics, and optionally saves/plots histogram and FFT.
"""

import argparse
import sys
import os
import time
import socket
import datetime

import numpy as np
import matplotlib.pyplot as plt

# Import the ACE Remote Control client
sys.path.append(r"C:\Program Files (x86)\Analog Devices\ACE\Client")
import clr
clr.AddReference('AnalogDevices.Csa.Remoting.Clients')
import AnalogDevices.Csa.Remoting.Clients as adrc

# UI and context paths
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
                        disable_channels='0,2,3'):
        """Set filter, data format, data frame, and power-down channels."""
        fmap = {'Sinc1': '0', 'Sinc2': '1', 'Sinc6': '2'}
        code = fmap.get(filter_type, filter_type)
        self.client.set_ContextPath(CTX_DRIVER)
        self.client.NavigateToPath(UI_DRIVER)
        for ch in disable_channels.split(','):
            self.client.Run(
                f'Evaluation.Control.SetIntParameter("PWRDN_CH{ch}",1,-1)'
            )
        # Sinc-6 hardware filter on channel 1
        self.client.Run(
            f'Evaluation.Control.SetIntParameter("virtual-parameter-filter-ch1",{code},-1)'
        )
        # 4-parallel data format
        self.client.Run(
            'Evaluation.Control.SetIntParameter("virtual-parameter-data-format",2,-1)'
        )
        # 24-bit no CRC: data frame = 2
        self.client.Run(
            'Evaluation.Control.SetIntParameter("virtual-parameter-data-frame",2,-1)'
        )
        # apply all settings
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
        # perform raw capture under driver context
        self.client.AsyncRawCaptureToFile(bin_path, 'capture', 'false', 'true')
        self.client.WaitOnRawCaptureToFile(str(timeout_ms), 'capture', 'false', bin_path)
        return bin_path


def read_raw_samples(bin_file, big_endian, scale):
    """
    Read 3-byte-packed samples from bin_file, sign-extend to 32 bits,
    and return a float array in volts.
    """
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
    # save CSV
    hist_csv = None
    if out_file:
        hist_csv = out_file.replace('.png', '.csv')
        np.savetxt(hist_csv,
                   np.vstack((edges[:-1], counts)).T,
                   delimiter=',',
                   header='bin_start,count',
                   comments='')
    # plot
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


def plot_fft(raw, out_file=None, show=False):
    N = raw.size
    freqs = np.fft.rfftfreq(N, d=1.0)
    fft_vals = np.abs(np.fft.rfft(raw))
    # save CSV
    fft_csv = None
    if out_file:
        fft_csv = out_file.replace('.png', '.csv')
        np.savetxt(fft_csv,
                   np.vstack((freqs, fft_vals)).T,
                   delimiter=',',
                   header='frequency,value',
                   comments='')
    # plot
    plt.figure()
    plt.plot(freqs, fft_vals)
    plt.xlabel('Normalized Frequency'); plt.ylabel('Magnitude')
    plt.title('FFT')
    plt.tight_layout()
    if out_file:
        plt.savefig(out_file)
    if show:
        plt.show()
    return fft_csv


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    nf = sub.add_parser('noise-floor')
    nf.add_argument('-n','--samples', type=int, default=131072,
                    help='Sample count')
    nf.add_argument('--ace-host', type=str, default='localhost:2357',
                    help='ACE host:port')
    nf.add_argument('--scale', type=float, default=4.8828125e-07,
                    help='Volts per count')
    nf.add_argument('--big-endian', action='store_true',
                    help='Interpret raw data as big-endian 24-bit words')
    nf.add_argument('--histogram', action='store_true',
                    help='Compute and plot histogram')
    nf.add_argument('--hist-bins', type=int, default=100,
                    help='Number of bins for histogram')
    nf.add_argument('--fft', action='store_true',
                    help='Compute and plot FFT')
    nf.add_argument('--plot', action='store_true',
                    help='Save raw plot')
    nf.add_argument('--show', action='store_true',
                    help='Immediately display plots')
    args = p.parse_args()

    if args.cmd == 'noise-floor':
        ace = ACEClient(args.ace_host)
        print(f"ACE @ {args.ace_host}, IP {ace.get_local_ip()}")
        ace.configure_board()
        bin_file = ace.capture(args.samples)
        print(f"Binary saved: {bin_file}")

        raw = read_raw_samples(bin_file, args.big_endian, args.scale)
        # metrics
        mean = raw.mean(); std = raw.std(); ptp = np.ptp(raw)
        print("\nResults:")
        print(f"  mean_V: {mean:.6e} V")
        print(f"  std_V : {std:.6e} V")
        print(f"  pp_V  : {ptp:.6e} V")

        # raw plot
        if args.plot or args.show:
            png = None
            if args.plot:
                png = bin_file.replace('.bin', '_raw.png')
            plot_raw(raw, out_file=png, show=args.show)
            if png:
                print(f"Raw plot saved: {png}")
            if args.show and not args.plot:
                print("Raw plot displayed")

        # histogram
        if args.histogram:
            h_png = None
            if args.plot:
                h_png = bin_file.replace('.bin', '_hist.png')
            h_csv = plot_histogram(raw, args.hist_bins, out_file=h_png, show=args.show)
            if h_png:
                print(f"Histogram plot saved: {h_png}")
            if h_csv:
                print(f"Histogram data saved: {h_csv}")
            if args.show and not args.plot:
                print("Histogram displayed")

        # FFT
        if args.fft:
            f_png = None
            if args.plot:
                f_png = bin_file.replace('.bin', '_fft.png')
            f_csv = plot_fft(raw, out_file=f_png, show=args.show)
            if f_png:
                print(f"FFT plot saved: {f_png}")
            if f_csv:
                print(f"FFT data saved: {f_csv}")
            if args.show and not args.plot:
                print("FFT displayed")


if __name__ == '__main__':
    main()
