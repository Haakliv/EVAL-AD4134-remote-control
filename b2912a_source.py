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

    # Perform a voltage sweep on the SMU
    # param start: Sweep start voltage in volts
    # param stop: Sweep stop voltage in volts
    # param points: Number of steps in the sweep (optional)
    # param step: Step size in volts (optional)
    # param current_limit: Compliance current in amps
    # param range_mode: 'AUTO', 'BEST', or 'FIXED'
    # param trigger_count: Number of sweeps to execute
    # param trigger_delay: Delay before starting sweep in seconds
    # param trigger_period: Interval between points in seconds (for TIMER trigger)
    def sweep_voltage(self,
                      start: float,
                      stop: float,
                      points: int = None,
                      step: float = None,
                      current_limit: float = 0.01,
                      range_mode: str = 'AUTO',
                      trigger_count: int = 1,
                      trigger_delay: float = 0.0,
                      trigger_period: float = None):
        # Configure source for sweep
        self.smu.write('SOUR:FUNC VOLT')
        self.smu.write(f'SENS:CURR:PROT {current_limit}')
        # Set sweep mode
        self.smu.write('SOUR:VOLT:MODE SWE')
        # Range operation for sweep
        self.smu.write(f'SOUR:VOLT:SWE:MODE {range_mode}')
        # Start/stop
        self.smu.write(f'SOUR:VOLT:SWE:STAR {start}')
        self.smu.write(f'SOUR:VOLT:SWE:STOP {stop}')
        # Points or step
        if points is not None:
            self.smu.write(f'SOUR:VOLT:SWE:POIN {points}')
        elif step is not None:
            self.smu.write(f'SOUR:VOLT:SWE:STEP {step}')
        else:
            raise ValueError('Either points or step must be specified')
        # Trigger settings
        self.smu.write(f'TRIG:COUN {trigger_count}')
        self.smu.write(f'TRIG:DEL {trigger_delay}')
        if trigger_period is not None:
            # Use TIMER trigger for fixed interval
            self.smu.write('TRIG:TYPE TIM')
            self.smu.write(f'TRIG:TIM {trigger_period}')
        else:
            # Default to immediate trigger
            self.smu.write('TRIG:TYPE IMM')
        # Start the sweep
        self.smu.write('INIT')

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
