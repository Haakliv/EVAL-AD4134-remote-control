import numpy as np
import matplotlib.pyplot as plt

AFE_gain = 1
ODR = 1.25e6

# Log-spaced frequency vector from 400 Hz to 625 kHz
frequencies_hz = np.logspace(np.log10(400), np.log10(625e3), 10000)

# Frequency response of sinc6 filter (linear)
gain_response = AFE_gain * (np.sinc(frequencies_hz / ODR))**6
gain_response_db = 20 * np.log10(gain_response)

# Find the frequency at the -3 dB point
idx_3dB = np.argmin(np.abs(gain_response_db - (-3)))
freq_3dB_hz = frequencies_hz[idx_3dB]

plt.figure(figsize=(8, 6))
plt.semilogx(frequencies_hz, gain_response_db, label='AC Gain [dB]')
plt.axhline(-3, color='red', linestyle='--', label='-3 dB level')
plt.axvline(freq_3dB_hz, color='green', linestyle='--', label=f'-3 dB freq = {freq_3dB_hz:.0f} Hz')

plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain [dB]')
plt.title('Theoretical Frequency Response (sinc6 Filter, dB Scale)')
plt.ylim(-25, 7)
plt.legend()
plt.grid(True, which='both', linestyle=':')
plt.tight_layout()
plt.show()

print(f"The -3 dB cutoff frequency is {freq_3dB_hz:.0f} Hz.")
