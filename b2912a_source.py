#!/usr/bin/env python3
import pyvisa as visa

class B2912A:
    # Agilent/Keysight B2912A SMU driver over LAN (TCPIP) using PyVISA. Connect via VISA resource string: 'TCPIP0::<IP>::inst0::INSTR'
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
    def set_voltage(self, voltage: float, current_limit: float = 0.01, range_mode: str = 'AUTO'):
        # Set source mode to voltage
        self.smu.write('SOUR:FUNC VOLT')
        # Set compliance (current limit)
        self.smu.write(f'SENS:CURR:PROT {current_limit}')
        # Set source range
        self.smu.write(f'SOUR:VOLT:RANG {range_mode}')
        # Apply voltage
        self.smu.write(f'SOUR:VOLT {voltage}')

    # Enable the SMU output
    def output_on(self):
        self.smu.write('OUTP ON')

    # Disable the SMU output
    def output_off(self):
        self.smu.write('OUTP OFF')

    # Turn off the output and close the VISA session
    def close(self):
        try:
            self.output_off()
        except Exception:
            pass
        self.smu.close()
        self.rm.close()
