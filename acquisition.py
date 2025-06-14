import numpy as np
import os
import time
import shutil
from common import ADC_SCALE

# Read raw binary samples from the AD4134 capture file and convert to scaled voltages.
def read_raw_samples(bin_file, scale=ADC_SCALE):
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
def capture_samples(
    ace_client,
    sample_count: int = 131072,
    scale: float = ADC_SCALE,
    timeout_ms: int = 10000,
    output_dir: str = None,
) -> np.ndarray:
    ace = ace_client
    bin_path = ace.capture(sample_count, timeout_ms)
    capture_samples.last_bin_path = bin_path

    # Move all capture artifacts into output_dir
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        folder = os.path.dirname(bin_path)
        prefix = os.path.basename(folder)
        for fname in os.listdir(folder):
            src = os.path.join(folder, fname)
            dst = os.path.join(output_dir, f"{prefix}_{fname}")
            shutil.move(src, dst)
        try:
            os.rmdir(folder)
        except OSError:
            pass
        bin_path = os.path.join(output_dir, f"{prefix}_{os.path.basename(bin_path)}")

    # Convert raw binary to voltage array
    return read_raw_samples(bin_path, scale)

# Initialize attribute so callers can always access the last .bin path
capture_samples.last_bin_path = None
