from zolve_instruments import Sdg6022x
from ace_client import limit_vpp_offset, MAX_INPUT_RANGE

# Wrapper for Siglent SDG6022X to configure and control output waveforms. Supports sine and pulse with safe amplitude limiting.
class WaveformGenerator:
    def __init__(self, host='192.168.1.100'):
        self.sdg = Sdg6022x(host)
        self.disable(1)
        self.disable(2)

    # Disable output on the specified channel
    def disable(self, channel):
        self.sdg.disable_output(channel)


    # Generate a sine wave
    # param channel: Output channel (1 or 2)
    # param frequency: Frequency in Hz
    # param amplitude: Peak-to-peak voltage in Vpp
    # param offset: DC offset in V
    def sine(self, channel, frequency, amplitude, offset):
        max_in = MAX_INPUT_RANGE
        if abs(offset) > max_in:
            old = offset
            offset = max_in if offset > 0 else -max_in
            print(f"Warning: offset {old}V outside ±{max_in}V, clamped to {offset}V")

        safe_vpp = limit_vpp_offset(amplitude, offset, max_input=max_in)
        if safe_vpp <= 0:
            raise ValueError(
                f"Offset={offset} V already at rail ±{max_in} V—no headroom for any Vpp!"
            )

        self.sdg.set_waveform('SINE', channel)
        self.sdg.set_frequency(frequency, channel)
        self.sdg.set_amplitude(safe_vpp, channel)
        self.sdg.set_offset(offset, channel)
        self.sdg.set_output_load('50', channel)
        self.sdg.enable_output(channel)

    # ------------------------------------------------------------------
    # Differential pulse: CH1 normal, CH2 inverted, outputs enabled last
    # ------------------------------------------------------------------
    def pulse_diff(
        self,
        frequency: float,
        amplitude: float,
        offset: float,
        low_pct: float = 80.0,     # 4 ms low, 1 ms high at 200 Hz
        edge_time: float = 2e-9,
        ch_pos: int = 1,
        ch_neg: int = 2,
    ):
        max_in = MAX_INPUT_RANGE*2
        if abs(offset) > max_in:
            offset = max_in if offset > 0 else -max_in

        safe_vpp = limit_vpp_offset(amplitude, offset)
        if safe_vpp <= 0:
            raise ValueError("Offset too close to rail for given Vpp")

        high_pct = 100.0 - low_pct  # SDG’s DUTY = %HIGH

        for ch in (ch_pos, ch_neg):
            self.sdg.set_waveform("PULSE", ch)
            self.sdg.set_frequency(frequency, ch)
            self.sdg.set_amplitude(safe_vpp, ch)
            self.sdg.set_offset(offset, ch)

            # correct 50 Ω load, error in library
            self.sdg.interface.write(f"C{ch}:OUTP LOAD,50")

            # pulse specifics
            self.sdg.interface.write(f"C{ch}:BSWV DUTY,{high_pct}")
            self.sdg.interface.write(f"C{ch}:BSWV DLY,0")
            self.sdg.interface.write(f"C{ch}:BSWV RISE,{edge_time}")
            self.sdg.interface.write(f"C{ch}:BSWV FALL,{edge_time}")

        # invert negative leg before outputs go ON
        self.sdg.interface.write(f"C{ch_neg}:OUTP PLRT,INVT")

        # enable outputs
        self.sdg.enable_output(ch_pos)
        self.sdg.enable_output(ch_neg)
