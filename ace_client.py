import os
import sys
import time
import socket
import datetime
import glob

# Add ACE Remote Control client library path
sys.path.append(r"C:\Program Files (x86)\Analog Devices\ACE\Client")
import clr
clr.AddReference('AnalogDevices.Csa.Remoting.Clients')
import AnalogDevices.Csa.Remoting.Clients as adrc

# Hardware-specific mappings for AD4134:
SLAVE_ODR_MAP = {
    0: 1_449_275.0,      # 1.449275 MHz
    1: 1_250_000.0,      # 1.25 MHz
    2: 1_123_596.0,      # 1.123596 MHz
    3: 800_000.0,        # 800 kHz
    4: 751_880.0,        # 751.88 kHz
    5: 645_161.0,        # 645.161 kHz
    6: 600_000.0,        # 600 kHz
    7: 363_636.0,        # 363.636 kHz
    8: 320_513.0,        # 320.513 kHz
    9: 250_000.0,        # 250 kHz
    10: 100_000.0,       # 100 kHz
    11: 50_000.0,        # 50 kHz
    12: 10_000.0,        # 10 kHz
    13: 1_000.0,         # 1 kHz
}

SINC_FILTER_MAP = {
    0: 'Sinc3',
    1: 'Sinc3 50Hz & 60Hz Rejection',
    2: 'Sinc6',
    3: 'Wideband01',
    4: 'Wideband04',
}


# UI & context paths for AD4134
UI_ROOT     = r"Root::System"
UI_BOARD    = UI_ROOT + r".Subsystem_1.AD4134 Eval Board"
UI_DRIVER   = UI_BOARD + r".AD4134"
UI_ANALYSIS = UI_DRIVER + r".AD4134 Analysis"

CTX_BOARD    = r"\\System\\Subsystem_1\\AD4134 Eval Board"
CTX_DRIVER   = CTX_BOARD + r"\\AD4134"
CTX_ANALYSIS = CTX_DRIVER + r"\\AD4134 Analysis"

ACE_PLUGIN = 'AD4134 Eval Board'
ADC_RES_BITS    = 24     # 24-bit ADC resolution
MAX_INPUT_RANGE = 4.096  # ±4.096 V input range of AD4134


# Ensure that requested peak-to-peak voltage and offset do not exceed input range.
# Returns a safe Vpp value (<= 2*(max_input - abs(offset))).
def limit_vpp_offset(requested_vpp, offset):
    allowed_vpp = 2 * (MAX_INPUT_RANGE - abs(offset))
    if requested_vpp > allowed_vpp:
        print(
            f"Warning: requested Vpp={requested_vpp} Vpp with offset={offset} V"
            f" exceeds input range ±{MAX_INPUT_RANGE} V. Limiting to {allowed_vpp:.4f} Vpp."
        )
        return allowed_vpp
    return requested_vpp


# Create a timestamped Measurements folder for saving capture files.
def _measurement_folder():
    base = os.path.dirname(os.path.abspath(__file__))
    meas_root = os.path.join(base, 'Measurements')
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = os.path.join(meas_root, ts)
    os.makedirs(folder, exist_ok=True)
    return folder


# Wrapper for ACE Remote Control for AD4134 using the Control API.
class ACEClient:
    def __init__(self, host='localhost:2357'):
        print(f"Connecting to ACE server at {host}...")
        mgr = adrc.ClientManager.Create()
        self.client = mgr.CreateRequestClient(host)
        print(f"Connected to ACE server, initializing board...")
        # Initialize board
        self.client.NavigateToPath(UI_ROOT)
        self.client.AddHardwarePlugin(ACE_PLUGIN)
        self.client.set_ContextPath(CTX_BOARD)
        self.client.NavigateToPath(UI_BOARD)
        self.client.Run('@Reset()')
        time.sleep(1)
        # Initialize driver
        self.client.set_ContextPath(CTX_DRIVER)
        self.client.NavigateToPath(UI_DRIVER)
        self.client.Run('@Initialization()')
        self.client.Run('@GetStatusFromBoard()')

    # Return the ACE server's IP address.
    def get_local_ip(self):
        return socket.gethostbyname(socket.gethostname())

    # Configure the ADC board with filter, data format, and power-down channels.
    # param filter_code: ADC filter code (0-4)
    # param disable_channels: CSV of channel indices to power down
    def configure_board(self, filter_code: int = 2, disable_channels='0,2,3'):
        print(f"Configuring board with filter {SINC_FILTER_MAP[filter_code]}")
        # filter_code is passed directly
        code = str(filter_code)
        disabled = [ch.strip() for ch in disable_channels.split(',') if ch.strip() != '']
        all_channels = ['0', '1', '2', '3']
        enabled_channels = [ch for ch in all_channels if ch not in disabled]
        enabled_ch = enabled_channels[0]

        self.client.set_ContextPath(CTX_DRIVER)
        self.client.NavigateToPath(UI_DRIVER)
        # Power-down channels
        for ch in disable_channels.split(','):
            self.client.Run(
                f'Evaluation.Control.SetIntParameter("PWRDN_CH{ch}",1,-1)'
            )
        # High performance mode
        self.client.Run('Evaluation.Control.SetIntParameter("virtual-parameter-power-mode",1,-1)')
        # Filter and data format settings
        self.client.Run(
            f'Evaluation.Control.SetIntParameter("virtual-parameter-filter-ch{enabled_ch}",{code},-1)'
        )
        self.client.Run('Evaluation.Control.SetIntParameter("virtual-parameter-data-format",2,-1)')
        self.client.Run('Evaluation.Control.SetIntParameter("virtual-parameter-data-frame",2,-1)')
        self.client.Run('@ApplySettings()')

    def setup_capture(self, sample_count: int = 131072, odr_code: int = 12):
        # Set sample count and ODR
        self.client.set_ContextPath(CTX_DRIVER)
        self.client.NavigateToPath(UI_DRIVER)
        self.client.Run(
            f'Evaluation.Control.SetIntParameter("virtual-parameter-sample-count",{sample_count},-1)'
        )
        self.client.Run(
            f'Evaluation.Control.SetIntParameter("virtual-parameter-slave-odr",{odr_code},-1)'
        )

    # Perform an asynchronous raw data capture to a binary file.
    # param sample_count: number of samples to capture
    # param odr_code: desired output data rate in Hz
    # param timeout_ms: timeout in milliseconds
    # return: path to the binary output file
    def capture(self, sample_count, timeout_ms=10000):
        # Create output folder & filename
        folder = _measurement_folder()
        bin_path = os.path.join(folder, f"raw_{sample_count}.bin")

        self.client.AsyncRawCaptureToFile(bin_path, 'capture', 'false', 'true')
        self.client.WaitOnRawCaptureToFile(
            str(timeout_ms), 'capture', 'false', bin_path
        )

        return bin_path
    