import os
import sys
import time
import socket
import datetime
import clr
import AnalogDevices.Csa.Remoting.Clients as ClientsModule
from common import SINC_FILTER_MAP, MAX_INPUT_RANGE

# Add ACE Remote Control client library path
sys.path.append(r"C:\Program Files (x86)\Analog Devices\ACE\Client")
clr.AddReference('AnalogDevices.Csa.Remoting.Clients')

# UI & context paths for AD4134
UI_ROOT     = r"Root::System"
UI_BOARD    = UI_ROOT + r".Subsystem_1.AD4134 Eval Board"
UI_DRIVER   = UI_BOARD + r".AD4134"
UI_ANALYSIS = UI_DRIVER + r".AD4134 Analysis"

CTX_BOARD    = r"\\System\\Subsystem_1\\AD4134 Eval Board"
CTX_DRIVER   = CTX_BOARD + r"\\AD4134"
CTX_ANALYSIS = CTX_DRIVER + r"\\AD4134 Analysis"

ACE_PLUGIN = 'AD4134 Eval Board'


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
        mgr = ClientsModule.ClientManager.Create()
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

    @staticmethod
    def get_local_ip():
        return socket.gethostbyname(socket.gethostname())

    # Configure the ADC board with filter, data format, and power-down channels.
    def configure_board(self, filter_code: int = 2, disable_channels='0,2,3'):
        print(f"Configuring board with filter {SINC_FILTER_MAP[filter_code]}")
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
    def capture(self, sample_count, timeout_ms=10000):
        # Create output folder & filename
        folder = _measurement_folder()
        bin_path = os.path.join(folder, f"raw_{sample_count}.bin")

        self.client.AsyncRawCaptureToFile(bin_path, 'capture', 'false', 'true')
        self.client.WaitOnRawCaptureToFile(
            str(timeout_ms), 'capture', 'false', bin_path
        )

        return bin_path
    