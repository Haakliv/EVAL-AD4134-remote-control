import numpy as np
from scipy.signal import find_peaks

# Compute the FFT magnitude spectrum of a real-valued signal.
# param raw: 1D numpy array of time-domain samples
# param fs: Sampling rate in Hz
# return: freqs (Hz), spectrum (magnitude)
def fft_spectrum(raw, fs):
    N = raw.size
    freqs = np.fft.rfftfreq(N, d=1.0/fs)
    spectrum = np.abs(np.fft.rfft(raw))
    return freqs, spectrum


# Calculate noise floor metrics: mean, standard deviation, and peak-to-peak.
# param raw: 1D numpy array of voltage samples
# return: (mean, std, ptp)
def compute_noise_floor_metrics(raw):
    mean = raw.mean()
    std = raw.std()
    ptp = np.ptp(raw)
    return mean, std, ptp


# Calculate SFDR, THD, SINAD, and ENOB for a single-tone input.
# param freqs: frequency bins (Hz)
# param spectrum: magnitude spectrum
# param f0: fundamental frequency (Hz)
# param num_harmonics: number of harmonics to include for THD
# return: (sfdr_dB, thd_dB, sinad_dB, enob_bits)
def compute_metrics(freqs, spectrum, f0, num_harmonics=5):
    # Fundamental power
    idx_f = np.argmin(np.abs(freqs - f0))
    P1 = spectrum[idx_f] ** 2

    # Power excluding DC
    spec_nd = spectrum.copy()
    spec_nd[0] = 0
    P_total = np.sum(spec_nd ** 2)
    P_noise_dist = P_total - P1

    sinad = 10 * np.log10(P1 / P_noise_dist)
    enob = (sinad - 1.76) / 6.02

    # THD: sum power of first `num_harmonics` harmonics
    P_harm = 0.0
    for n in range(2, num_harmonics + 1):
        idx_h = np.argmin(np.abs(freqs - n * f0))
        P_harm += spectrum[idx_h] ** 2
    thd = 10 * np.log10(P_harm / P1)

    # SFDR: exclude fundamental, find largest spur
    spec_spurs = spec_nd.copy()
    spec_spurs[idx_f] = 0
    max_spur = np.max(spec_spurs)
    sfdr = 20 * np.log10(spectrum[idx_f] / max_spur)

    return sfdr, thd, sinad, enob


import numpy as np

# --------------------------------------------------------------------------
def compute_settling_time(
    raw: np.ndarray,
    fs: float,
    tol_uV: float,
):
    """
    Simple sample‐based settling‐time detection:
      - Edge: first rising step > tol
      - Settling: first of three consecutive samples within tol of final value
    Returns (settling_start_idx, settling_end_idx, Ts_us, segment)
    or (None, None, None, None) if not found.
    """
    Ts_us = 1e6 / fs
    thr   = tol_uV * 1e-6
    settling_start = None
    settling_end   = None

    # 1) find rising edge
    for i in range(1, len(raw)):
        if raw[i] - raw[i - 1] > thr:
            settling_start = i-1
            break
    if settling_start is None:
        return None, None, None, None

    # 2) find falling edge (to avoid using that region for final level)
    fall_idxs = np.where(np.diff(raw) < -thr)[0]
    if fall_idxs.size > 0:
        idx_fall = fall_idxs[0] + 1
    else:
        idx_fall = len(raw)

    # 3) compute the “final” plateau level from the 3 samples just before idx_fall
    start_plateau = max(settling_start, idx_fall - 3)
    final_val = float(np.mean(raw[start_plateau:idx_fall]))

    # 4) look for first run of 3 consecutive samples within tol of final_val
    for j in range(settling_start + 1, idx_fall - 2):
        window = raw[j : j + 3]
        if np.all(np.abs(window - final_val) < thr):
            settling_end = j
            break

    # 5) return indices, sample period, and the raw segment for inspection
    segment = raw[settling_start : settling_end]
    return settling_start, settling_end, Ts_us, segment

# Compute the -db_down dB bandwidth from a gain sweep.
# param freqs: array of stimulus frequencies (Hz)
# param gains: corresponding linear gains
# param db_down: dB drop from reference gain (default 3 dB)
# return: bandwidth frequency (Hz)
def compute_bandwidth(freqs, gains, db_down=3):
    ref_gain = gains[0]
    cutoff_lin = ref_gain / (10 ** (db_down / 20))
    for f, g in zip(freqs, gains):
        if g <= cutoff_lin:
            return f
    return freqs[-1]


# Compute DC gain and offset errors from test data.
# param applied: sequence of nominal voltages applied by SMU
# param verified: sequence of actual voltages measured by DMM (or None)
# param adc: sequence of voltages read by the ADC
# return: dict with keys 'gain', 'offset', and fit R^2
def compute_dc_gain_offset(applied, verified, adc):
    # Use the verified values if provided, else use applied
    x = np.array(verified) if verified[0] is not None else np.array(applied)
    y = np.array(adc)
    # Linear fit y = m*x + b
    m, b = np.polyfit(x, y, 1)
    # Goodness of fit
    y_pred = m * x + b
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res/ss_tot
    return {'gain': m, 'offset': b, 'r2': r2}

def find_spur_rms(freqs, spectrum, target_freq, span_hz=5e3):
    """
    Locate and return the RMS voltage (in volts) of the spur nearest target_freq.
    
    - freqs: array of FFT bin centers [Hz]
    - spectrum: magnitude spectrum (peak values) [V_peak]
    - target_freq: spur center frequency [Hz]
    - span_hz: hunt ± this window around target_freq [Hz]
    
    Returns (spur_rms, actual_freq), where spur_rms = V_peak/√2, actual_freq = bin freq.
    """
    # 1) limit to a window around the target
    mask = np.abs(freqs - target_freq) <= span_hz
    idxs = np.nonzero(mask)[0]
    if idxs.size == 0:
        raise ValueError(f"No bins within ±{span_hz} Hz of {target_freq} Hz")
    
    # 2) find peaks in that window
    sub = spectrum[idxs]
    peaks, _ = find_peaks(sub, height=np.max(sub)*0.1)  # only peaks >10% of local max
    if peaks.size == 0:
        # fallback to the single largest bin
        peak_idx = idxs[np.argmax(sub)]
    else:
        # pick the largest detected peak
        peak_idx = idxs[peaks[np.argmax(sub[peaks])]]
    
    # 3) compute RMS from peak value
    v_peak  = spectrum[peak_idx]
    v_rms   = v_peak / np.sqrt(2)
    return v_rms, freqs[peak_idx]
