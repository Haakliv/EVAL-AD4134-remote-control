#!/usr/bin/env python3
"""
b2912a_source.py

Agilent B2912A SMU driver over LAN (TCPIP) using PyVISA.

Connect via VISA resource string: 'TCPIP0::<IP>::inst0::INSTR'
"""
import pyvisa as visa

class B2912A:
    """
    Wrapper for Agilent/Keysight B2912A SMU over Ethernet.
    """
    def __init__(self, resource: str):
        """
        :param resource: VISA address of the SMU, e.g. 'TCPIP0::192.168.1.100::inst0::INSTR'
        """
        self.rm = visa.ResourceManager()
        self.smu = self.rm.open_resource(resource)
        # Reset and clear status
        self.smu.write('*RST')
        self.smu.write('*CLS')

    def set_voltage(self, voltage: float, current_limit: float = 0.01, range_mode: str = 'AUTO'):
        """
        Configure DC voltage source on the SMU.

        :param voltage: Output voltage in volts
        :param current_limit: Compliance current in amps
        :param range_mode: 'AUTO', 'BEST', or 'FIXED'
        """
        # Set source mode to voltage
        self.smu.write('SOUR:FUNC VOLT')
        # Set compliance (current limit)
        self.smu.write(f'SENS:CURR:PROT {current_limit}')
        # Set source range
        self.smu.write(f'SOUR:VOLT:RANG {range_mode}')
        # Apply voltage
        self.smu.write(f'SOUR:VOLT {voltage}')

    def output_on(self):
        """Enable the SMU output"""
        self.smu.write('OUTP ON')

    def output_off(self):
        """Disable the SMU output"""
        self.smu.write('OUTP OFF')

    def close(self):
        """
        Turn off output and close the VISA session.
        """
        try:
            self.output_off()
        except Exception:
            pass
        self.smu.close()
        self.rm.close()
