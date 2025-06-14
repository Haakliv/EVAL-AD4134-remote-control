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
        
        self.smu.write(':SENS1:REM ON') # 4W mode
        self.smu.write(":SENS1:CURR:PROT 0.001") # 1mA current protection
        self.smu.write(':SENS1:CURR:RANG 0.001') # 1mA current range
        self.smu.write(':SOUR1:VOLT:RANG:AUTO OFF') # 20V range
        self.smu.write(':SOUR1:VOLT:RANG 20') # 20V range
        self.smu.write(':SENS1:VOLT:RANG:AUTO OFF') # 20V range
        self.smu.write(':SENS1:VOLT:RANG 20') # 20V range
        self.smu.write(':OUTP1:FILT:AUTO OFF') # Disable auto filter
        self.smu.write(':OUTP1:FILT:LPAS:STAT ON') # Low-pass filter ON
        self.smu.write(':OUTP1:FILT:LPAS:FREQ MIN') # Minimum low-pass filter frequency
        self.smu.write(':OUTP1:HCAP 1') # High capacitance output filter

    # Configure DC voltage source on the SMU
    def set_voltage(self, voltage):
        allowed_vpp = limit_vpp_offset(requested_vpp=0.0,
                                       offset=voltage)
        if allowed_vpp < 0:
            safe_v = MAX_INPUT_RANGE if voltage >= 0 else -MAX_INPUT_RANGE
        else:
            safe_v = voltage

        self.smu.write(f'SOUR:VOLT {safe_v}')

    def measure_voltage(self) -> float:
        return float(self.smu.query(':MEAS:VOLT:DC?'))

    def set_voltage_blocking(self, voltage):
        self.smu.write("*OPC")  # Arm the operation complete flag
        self.set_voltage(voltage)
        self.smu.query("*OPC?")  # Wait for completion

    def output_on(self):
        self.smu.write('OUTP ON')

    def output_off(self):
        self.smu.write('OUTP OFF')

    def write(self, command: str):
        self.smu.write(command)

    def close(self):
        try:
            self.output_off()
        except Exception:
            pass
        self.smu.close()
        self.rm.close()
