from zolve_instruments import Sdg6022x
from ace_client import limit_vpp_offset, MAX_INPUT_RANGE

# Wrapper for Siglent SDG6022X to configure and control output waveforms. Supports sine and pulse with safe amplitude limiting.
class WaveformGenerator:
    offset = 0.0

    def __init__(self, host='192.168.1.100', waveform='SINE', offset=0.0):
        max_in = MAX_INPUT_RANGE*2

        if abs(offset) > max_in:
            offset = max_in if offset > 0 else -max_in
        self.offset = offset

        self.sdg = Sdg6022x(host)
        self.disable(1)
        self.disable(2)
        self.sdg.interface.write("C1:MODE PHASE-LOCKED")
        self.sdg.interface.write("C2:OUTP PLRT,INVT")
        self.sdg.set_waveform(waveform, 1)
        self.sdg.set_waveform(waveform, 2)
        self.sdg.set_offset(offset, 1)
        self.sdg.set_offset(offset, 2)

    def disable(self, channel):
        self.sdg.disable_output(channel)

    def enable(self, channel):
        self.sdg.enable_output(channel)

    def pulse_diff(
        self,
        frequency: float,
        amplitude: float,
        low_pct: float = 80.0, # 4 ms low, 1 ms high at 200 Hz
        edge_time: float = 2e-9,
        ch_pos: int = 1,
        ch_neg: int = 2,
    ):

        safe_vpp = limit_vpp_offset(amplitude, self.offset)
        if safe_vpp <= 0:
            raise ValueError("Offset too close to rail for given Vpp")

        high_pct = 100.0 - low_pct  # SDG’s DUTY = %HIGH

        for ch in (ch_pos, ch_neg):
            self.sdg.set_frequency(frequency, ch)
            self.sdg.set_amplitude(safe_vpp, ch)

            # 50 ohm load
            self.sdg.interface.write(f"C{ch}:OUTP LOAD,50")

            self.sdg.interface.write(f"C{ch}:BSWV DUTY,{high_pct}")
            self.sdg.interface.write(f"C{ch}:BSWV DLY,0")
            self.sdg.interface.write(f"C{ch}:BSWV RISE,{edge_time}")
            self.sdg.interface.write(f"C{ch}:BSWV FALL,{edge_time}")

        self.sdg.interface.write(f"C{ch_neg}:OUTP PLRT,INVT")

        self.sdg.enable_output(ch_pos)
        self.sdg.enable_output(ch_neg)

    def sine_diff(self,
                  frequency: float,
                  amplitude: float,
                  offset: float,
                  ch_pos: int = 1,
                  ch_neg: int = 2):
        max_in = MAX_INPUT_RANGE*2
        if abs(offset) > max_in:
            offset = max_in if offset > 0 else -max_in

        safe_vpp = limit_vpp_offset(amplitude, offset)
        if safe_vpp <= 0:
            raise ValueError(
                f"Offset={offset} V leaves no headroom for Vpp={amplitude} V"
            )

        self.sdg.set_frequency(frequency, ch_pos)
        self.sdg.set_frequency(frequency, ch_neg)
        self.sdg.set_amplitude(safe_vpp, ch_pos)
        self.sdg.set_amplitude(safe_vpp, ch_neg)
