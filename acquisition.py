import numpy as np
from ace_client import ADC_RES_BITS, MAX_INPUT_RANGE, SLAVE_ODR_MAP
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

def capture_samples_continuous(
    ace_client,
    sweep_time: float,
    scale: float = ADC_SCALE,
    odr_code: int = 12,
    samples_per_segment: int = 131072,
    timeout_ms: int = 10000,
    output_dir: str = None,
    run_idx: int = 1,
) -> np.ndarray:
    """
    Capture repeated segments using ace.capture() until sweep_time has elapsed.
    Each run's files are stored in a unique subfolder: .../run_01/, .../run_02/, etc.
    Returns the voltage samples as a single array (trimmed to sweep_time).
    """
    ace = ace_client
    odr_hz = SLAVE_ODR_MAP[odr_code]
    t_start = time.time()
    all_segments = []

    main_folder = output_dir

    run_folder = os.path.join(main_folder, f"run_{run_idx:02d}")
    os.makedirs(run_folder, exist_ok=True)

    # Setup once for efficiency
    ace.setup_capture(samples_per_segment, odr_code)

    # Segment acquisition loop (run until sweep_time is up)
    while True:
        elapsed = time.time() - t_start
        if elapsed >= sweep_time:
            break

        # Capture segment
        bin_path = ace.capture(samples_per_segment, timeout_ms)
        capture_samples_continuous.last_bin_path = bin_path

        # Move files into run subfolder
        folder = os.path.dirname(bin_path)
        prefix = os.path.basename(folder)
        for fname in os.listdir(folder):
            src = os.path.join(folder, fname)
            dst = os.path.join(run_folder, f"{prefix}_{fname}")
            shutil.move(src, dst)
        try:
            os.rmdir(folder)
        except OSError:
            pass
        segment_file = os.path.join(run_folder, f"{prefix}_{os.path.basename(bin_path)}")

        # Decode segment and store
        samples = read_raw_samples(segment_file, scale)
        all_segments.append(samples)

    # Concatenate and trim to exact sweep duration
    all_samples = np.concatenate(all_segments)
    n_target = int(sweep_time * odr_hz)
    if all_samples.size > n_target:
        all_samples = all_samples[:n_target]

    return all_samples

capture_samples_continuous.last_bin_path = None