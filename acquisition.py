import numpy as np
from ace_client import ACEClient, ADC_RES_BITS, MAX_INPUT_RANGE
import os
import time


def read_raw_samples(bin_file, scale=(MAX_INPUT_RANGE / (2 ** (ADC_RES_BITS - 1)))):
    """
    Read raw binary captures from the AD4134 and convert to scaled voltages.

    :param bin_file: Path to the .bin file containing raw 3-byte samples
    :param scale: Multiplicative scale factor (e.g. LSB weight)
    :return: 1D numpy array of float voltages
    """
    # Wait for file to exist (with a timeout)
    deadline = time.time() + 5.0
    while time.time() < deadline and not os.path.isfile(bin_file):
        time.sleep(0.1)
    if not os.path.isfile(bin_file):
        raise FileNotFoundError(f"[read_raw_samples] file still not found: {bin_file!r}")

    # Read raw bytes
    with open(bin_file, 'rb') as f:
        raw_bytes = f.read()
    data = np.frombuffer(raw_bytes, dtype=np.uint8)

    # Reshape into samples of 3 bytes each
    samples = data.reshape(-1, 3)

    # Combine bytes into 24-bit words
    codes = (
        (samples[:, 2].astype(np.int32) << 16)
        | (samples[:, 1].astype(np.int32) << 8)
        | samples[:, 0].astype(np.int32)
    )

    # Sign-extend 24-bit to 32-bit
    raw_counts = (codes << 8) >> 8

    # Convert counts to voltage
    return raw_counts.astype(float) * scale


def capture_samples(
    ace_host: str = 'localhost:2357',
    sample_count: int = 131072,
    scale: float = 1.0,
    odr_code: int = 12,
    filter_code: int = 2,
    disable_channels: str = '0,2,3',
    timeout_ms: int = 10000
) -> np.ndarray:
    """
    Configure ADC board, perform capture, and return scaled samples.

    :param ace_host: Address of ACE remote control server
    :param sample_count: Number of samples to capture
    :param scale: LSB-to-voltage scale factor
    :param odr_code: Output data rate code (0–13)
    :param filter_code: ADC filter code (0–4)
    :param disable_channels: CSV list of channels to power-down
    :param timeout_ms: Capture timeout in milliseconds
    :return: numpy array of voltage samples
    """
    # Initialize and configure the board
    ace = ACEClient(ace_host)
    ace.configure_board(
        filter_code=filter_code,
        disable_channels=disable_channels,
        odr_code=odr_code
    )

    # Capture raw binary
    bin_path = ace.capture(sample_count, odr_code, timeout_ms)

    # Remember where we just wrote the .bin
    capture_samples.last_bin_path = bin_path

    # Change dir for plots
    folder = os.path.dirname(bin_path)
    os.chdir(folder)

    # Read and scale samples
    return read_raw_samples(bin_path, scale)


# Initialize attribute so callers can always access the last .bin path
capture_samples.last_bin_path = None
