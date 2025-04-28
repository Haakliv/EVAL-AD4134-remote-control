import numpy as np


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


# Estimate the settling time of a step response in the raw data.
# param raw: time-domain samples
# param fs: sampling rate (Hz)
# param threshold: absolute voltage threshold for settling
# return: settling time in seconds or None
def compute_settling_time(raw, fs, threshold):
    N = raw.size
    t = np.arange(N) / fs
    final = np.mean(raw[int(0.9 * N):])
    idx0 = N // 2
    within = np.abs(raw - final) <= threshold
    for i in range(idx0, N):
        if within[i] and np.all(within[i:]):
            return t[i] - t[idx0]
    return None


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
