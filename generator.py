from zolve_instruments import Sdg6022x
from ace_client import limit_vpp_offset

# Wrapper for Siglent SDG6022X to configure and control output waveforms. Supports sine and pulse with safe amplitude limiting.
class WaveformGenerator:
    def __init__(self, host='192.168.1.100'):
        self.sdg = Sdg6022x(host)

    # Generate a sine wave
    # param channel: Output channel (1 or 2)
    # param frequency: Frequency in Hz
    # param amplitude: Peak-to-peak voltage in Vpp
    # param offset: DC offset in V
    def sine(self, channel, frequency, amplitude, offset):
        safe_vpp = limit_vpp_offset(amplitude, offset)
        self.sdg.set_waveform('SINE', channel)
        self.sdg.set_frequency(frequency, channel)
        self.sdg.set_amplitude(safe_vpp, channel)
        self.sdg.set_offset(offset, channel)
        self.sdg.enable_output(channel)

    # Generate a pulse wave (for settling-time tests)
    # param channel: Output channel (1 or 2)
    # param frequency: Pulse repetition rate in Hz
    # param amplitude: Peak-to-peak voltage in Vpp
    # param offset: DC offset in V
    def pulse(self, channel, frequency, amplitude, offset):
        safe_vpp = limit_vpp_offset(amplitude, offset)
        self.sdg.set_waveform('PULSE', channel)
        self.sdg.set_frequency(frequency, channel)
        self.sdg.set_amplitude(safe_vpp, channel)
        self.sdg.set_offset(offset, channel)
        self.sdg.enable_output(channel)

    # Disable output on the specified channel
    def disable(self, channel):
        self.sdg.disable_output(channel)
