#!/usr/bin/env python3
import pyvisa as visa
from ace_client import limit_vpp_offset, MAX_INPUT_RANGE

class B2912A:
    # Agilent/Keysight B2912A SMU driver over LAN (TCPIP) using PyVISA.
    # Connect via VISA resource string: 'TCPIP0::<IP>::inst0::INSTR'
    def __init__(self, resource: str):
        # param resource: VISA address of the SMU, e.g. 'TCPIP0::169.254.5.2::inst0::INSTR'
        self.rm = visa.ResourceManager()
        self.smu = self.rm.open_resource(resource)
        self.smu.write('*RST')
        self.smu.write('*CLS')

    # Configure DC voltage source on the SMU
    # param voltage: Output voltage in volts
    # param current_limit: Compliance current in amps
    # param range_mode: 'AUTO', 'BEST', or 'FIXED'
    def set_voltage(self,
                    voltage: float,
                    current_limit: float = 0.01,
                    range_mode: str = 'AUTO'):
        # “0 Vpp” sine with this offset → see if it’s legal
        allowed_vpp = limit_vpp_offset(requested_vpp=0.0,
                                       offset=voltage)

        # If allowed_vpp < 0 then abs(offset) > max_input, so clamp
        if allowed_vpp < 0:
            safe_v = MAX_INPUT_RANGE if voltage >= 0 else -MAX_INPUT_RANGE
        else:
            safe_v = voltage

        # SCPI for static voltage
        self.smu.write('SOUR:FUNC VOLT')
        self.smu.write(f'SENS:CURR:PROT {current_limit}')
        self.smu.write(f'SOUR:VOLT:RANG {range_mode}')
        self.smu.write(f'SOUR:VOLT {safe_v}')

    def output_on(self):
        self.smu.write('OUTP ON')

    def output_off(self):
        self.smu.write('OUTP OFF')

    def close(self):
        try:
            self.output_off()
        except Exception:
            pass
        self.smu.close()
        self.rm.close()
