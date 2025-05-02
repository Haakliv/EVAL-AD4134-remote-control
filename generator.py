from zolve_instruments import Sdg6022x
from ace_client import limit_vpp_offset, MAX_INPUT_RANGE

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
        self.sdg.enable_output(channel)

    def pulse(
        self,
        channel: int,
        frequency: float,
        amplitude: float,
        offset: float,
        low_pct: float = 90.0,
        edge_time: float | None = None,
    ):
        """
        Step‑style pulse that *waits* at the LOW level before rising.

        Parameters
        ----------
        channel   : 1 | 2
        frequency : Hz – repetition rate
        amplitude : Vpp – peak‑to‑peak swing
        offset    : V   – DC offset
        low_pct   : %   – portion of each period held LOW (default 90 %)
        edge_time : s   – optional rise/fall time; omit for instrument default
        """
        safe_vpp = limit_vpp_offset(amplitude, offset)

        period    = 1.0 / frequency
        high_pct  = 100.0 - low_pct            # generator wants %HIGH
        delay_sec = period * (low_pct / 100.0)

        cmd = (
            f"C{channel}:BSWV "
            f"WVTP,PULSE,"
            f"FRQ,{frequency},"
            f"DUTY,{high_pct},"
            f"DLY,{delay_sec},"
            f"AMP,{safe_vpp},"
            f"OFST,{offset},"
            f"RISE,{edge_time},FALL,{edge_time}"
        )

        self.sdg.interface.write(cmd)
        self.sdg.enable_output(channel)


    # Disable output on the specified channel
    def disable(self, channel):
        self.sdg.disable_output(channel)
