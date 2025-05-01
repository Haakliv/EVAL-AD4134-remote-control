import numpy as np
from ace_client import ADC_RES_BITS, MAX_INPUT_RANGE
import os
import time
import shutil

ADC_SCALE = MAX_INPUT_RANGE / (2 ** (ADC_RES_BITS - 1))  # LSB weight for ±4.096 V input range

# Read raw binary samples from the AD4134 capture file and convert to scaled voltages.
# param bin_file: Path to the .bin file containing raw 3-byte samples
# param scale: Multiplicative scale factor (e.g. LSB weight)
# return: 1D numpy array of float voltages
def read_raw_samples(bin_file, scale=ADC_SCALE):
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


# Configure ADC board and perform capture, and return scaled samples.
# param ace_host: Address of ACE remote control server
# param sample_count: Number of samples to capture
# param scale: LSB-to-voltage scale factor
# param odr_code: Output data rate code (0–13)
# param filter_code: ADC filter code (0–4)
# param disable_channels: CSV list of channels to power-down
# param timeout_ms: Capture timeout in milliseconds
# return: numpy array of voltage samples
def capture_samples(
    ace_client,
    sample_count: int = 131072,
    scale: float = ADC_SCALE,
    odr_code: int = 12,
    timeout_ms: int = 10000,
    output_dir: str = None,
) -> np.ndarray:
    ace = ace_client
    # Perform capture; this creates a timestamped folder and files (.bin, .cso, etc.)
    bin_path = ace.capture(sample_count, odr_code, timeout_ms)
    capture_samples.last_bin_path = bin_path

    # Move all capture files into output_dir with unique prefixes
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        folder = os.path.dirname(bin_path)
        prefix = os.path.basename(folder)
        for fname in os.listdir(folder):
            src = os.path.join(folder, fname)
            dst_name = f"{prefix}_{fname}"
            dst = os.path.join(output_dir, dst_name)
            shutil.move(src, dst)
        # clean up the now-empty folder
        try:
            os.rmdir(folder)
        except OSError:
            pass
        # update bin_path to the moved file
        moved_name = f"{prefix}_{os.path.basename(bin_path)}"
        bin_path = os.path.join(output_dir, moved_name)

    return read_raw_samples(bin_path, scale)

# Initialize attribute so callers can always access the last .bin path
capture_samples.last_bin_path = None
