#!/usr/bin/env python3
"""
b2912a_source.py

Control Agilent B2912A SMU for DC/pulse/sweep outputs via VISA.
"""

import visa
import numpy as np
import csv
import argparse
import time


class B2912A:
    """Wrapper for Agilent/Keysight B2912A Source/Measure Unit."""
    def __init__(self, resource):
        """
        :param resource: VISA resource string, e.g. 'USB0::0x0957::0x8E18::MY123456::INSTR'
        """
        self.rm = visa.ResourceManager()
        self.smu = self.rm.open_resource(resource)
        self.smu.write('*RST')
        self.smu.write('*CLS')

    def set_voltage(self, voltage, current_limit=0.01, range_mode='AUTO'):
        """
        :param voltage: Vpp for DC output
        :param current_limit: compliance in A
        :param range_mode: 'AUTO', 'BEST', or 'FIXED'
        SCPI:
          SOUR:FUNC VOLT
          SOUR:VOLT:MODE FIXed|LIST|LIN|LOG
          SOUR:VOLT {voltage}
          SENS:CURR:PROT {current_limit}
          SOUR:VOLT:RANG {range_mode}
        """
        self.smu.write('SOUR:FUNC VOLT')
        self.smu.write(f'SENS:CURR:PROT {current_limit}')
        self.smu.write(f'SOUR:VOLT {voltage}')
        self.smu.write(f'SOUR:VOLT:RANG {range_mode}')

    def set_current(self, current, voltage_limit=20, range_mode='AUTO'):
        """
        :param current: DC current in A
        :param voltage_limit: compliance
        :param range_mode: 'AUTO', 'BEST', or 'FIXED'
        SCPI analogous to voltage mode.
        """
        self.smu.write('SOUR:FUNC CURR')
        self.smu.write(f'SENS:VOLT:PROT {voltage_limit}')
        self.smu.write(f'SOUR:CURR {current}')
        self.smu.write(f'SOUR:CURR:RANG {range_mode}')

    def output_on(self):
        """SCPI: OUTP ON"""
        self.smu.write('OUTP ON')

    def output_off(self):
        """SCPI: OUTP OFF"""
        self.smu.write('OUTP OFF')

    def pulse(self, voltage, current_limit, period, width, polarity='POS'):
        """
        :param period: total period in s
        :param width: pulse width in s
        :param polarity: 'POS' or 'NEG'
        SCPI:
          SOUR:FUNC PULS
          SOUR:PULS:TIM {width}
          SOUR:PULS:PER {period}
          SOUR:PULS:POL {polarity}
        """
        self.smu.write('SOUR:FUNC PULS')
        self.smu.write(f'SENS:CURR:PROT {current_limit}')
        self.smu.write(f'SOUR:PULS:TIM {width}')
        self.smu.write(f'SOUR:PULS:PER {period}')
        self.smu.write(f'SOUR:PULS:POL {polarity}')
        self.smu.write(f'SOUR:VOLT {voltage}')

    def sweep(self, start, stop, points=50, kind='LIN', mode='SING'):
        """
        :param start: start voltage
        :param stop: stop voltage
        :param points: number of steps
        :param kind: 'LIN' or 'LOG'
        :param mode: 'SING' or 'DOUB'
        SCPI:
          SOUR:VOLT:MODE {kind}
          SOUR:SWE:SPAC {LIN|LOG}
          SOUR:SWE:STAR {start}
          SOUR:SWE:STOP {stop}
          SOUR:SWE:POIN {points}
          SOUR:SWE:DIR {UP|DOWN}
          SOUR:SWE:MODE {SING|DOUB}
        """
        self.smu.write('SOUR:FUNC VOLT')
        self.smu.write(f'SOUR:VOLT:MODE SWEE')
        self.smu.write(f'SOUR:SWE:SPAC {kind}')
        self.smu.write(f'SOUR:SWE:STAR {start}')
        self.smu.write(f'SOUR:SWE:STOP {stop}')
        self.smu.write(f'SOUR:SWE:POIN {points}')
        self.smu.write(f'SOUR:SWE:MODE {mode}')

    def list_sweep_from_file(self, fname, channel=1):
        """
        Load a list of voltages from CSV/TXT/PRN and apply as sweep.
        SCPI: SOUR:VOLT:MODE LIST / SOUR:LIST:DATA ...
        """
        with open(fname, newline='') as f:
            data = [float(row[0]) for row in csv.reader(f)]
        # Build comma-separated list
        datalist = ','.join(f'{v}' for v in data)
        self.smu.write('SOUR:FUNC VOLT')
        self.smu.write('SOUR:VOLT:MODE LIST')
        self.smu.write(f'SOUR:LIST:DATA {datalist}')
        self.smu.write(f'SOUR:LIST:POIN {len(data)}')

    def measure(self):
        """
        Query both voltage and current:
        SCPI: MEAS:VOLT?; MEAS:CURR?
        """
        v = float(self.smu.query('MEAS:VOLT?'))
        i = float(self.smu.query('MEAS:CURR?'))
        return v, i

    def close(self):
        self.output_off()
        self.smu.close()
        self.rm.close()