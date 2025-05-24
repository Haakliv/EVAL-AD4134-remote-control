import numpy as np
import scipy.signal
from scipy.signal import find_peaks

def compute_settling_time(
    raw: np.ndarray,
    fs: float,
    tol_uV: float,
    final_offset_us: float = 6.0,
    final_duration_us: float = 3.0,
    min_stable_samples: int = 10,
):
    """
    Settling-time detection with:
      - 1x LSB edge detection
      - robust final-value estimation (after step)
      - N consecutive samples within tolerance settling condition
    """
    Ts_us = 1e6 / fs

    # 1 detect the step edge
    settling_start = None
    for i in range(1, len(raw)):
        if raw[i] - raw[i-1] > tol_uV:
            settling_start = i - 1
            break
    if settling_start is None:
        return None, None, None

    # 2 estimate the final value from a later window
    idx0 = int(settling_start + final_offset_us  / Ts_us)
    idx1 = int(idx0            + final_duration_us/ Ts_us)
    if idx1 > len(raw):
        return None, None, None
    final_val = float(raw[idx0:idx1].mean())

    # 3 find the first index i where the next N samples are all within ±thr of final_val
    settling_end = None
    last_start = len(raw) - min_stable_samples + 1
    for i in range(settling_start+1, last_start):
        block = raw[i : i + min_stable_samples]
        if np.all(np.abs(block - final_val) < tol_uV):
            # mark settling point at the end of the stable block
            settling_end = i
            break

    if settling_end is None:
        # never saw N in a row within tolerance
        return None, None, None

    return settling_start, settling_end, Ts_us

def compute_mean_settling_time(
    raw_runs: list[np.ndarray],
    start_idxs: list[int],
    end_idxs:   list[int],
    Ts_us:      float,
    pad:        int = 2
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Compute and return the mean settling trace, truncated to pad samples 
    before edge and pad samples after the settling end.
    Returns:
      mean_start_us, mean_end_us, mean_delta_us,
      time_vec_us (truncated), mean_segment (truncated)
    """
    # 1 per-run times
    t_starts = np.array(start_idxs) * Ts_us
    t_ends   = np.array(end_idxs)   * Ts_us
    deltas   = t_ends - t_starts

    mean_delta = deltas.mean()

    # 2 find equal-length window across runs
    pre_samps   = min(start_idxs)
    post_samps  = min(len(raw) - end for raw, end in zip(raw_runs, end_idxs))
    delta_samps = min(e - s for s, e in zip(start_idxs, end_idxs))

    total_len = pre_samps + delta_samps + post_samps

    aligned = []
    for raw, s_idx in zip(raw_runs, start_idxs):
        seg = raw[s_idx - pre_samps : s_idx - pre_samps + total_len]
        aligned.append(seg)
    aligned = np.vstack(aligned)

    # 3 full mean trace
    mean_trace = aligned.mean(axis=0)

    # 4 full time vector (us)
    time_full = (np.arange(-pre_samps, -pre_samps + total_len) * Ts_us)

    # 5 now truncate to pad around t=0 and t=mean_delta
    # find indices
    zero_idx = np.searchsorted(time_full, 0)
    end_idx  = np.searchsorted(time_full, mean_delta)
    start_i  = max(0, zero_idx - pad)
    stop_i   = min(len(time_full), end_idx + pad)

    time_trunc = time_full[start_i:stop_i]
    trace_trunc = mean_trace[start_i:stop_i]

    return mean_delta, time_trunc, trace_trunc

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
