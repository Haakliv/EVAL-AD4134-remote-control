import time
import numpy as np
from zolve_instruments import Dmm6500

class Dmm6500Controller:
    def __init__(self, ip_address, timeout_sec=5):
        self.dmm = Dmm6500(ip_address, timeout_sec)
        self.configure_for_precise_dc()

    def configure_for_precise_dc(self, *, nplc=5,
                                dmm_range=10):
        self.dmm.interface.write(":SENS:FUNC 'VOLT:DC'")
        self.dmm.interface.write(f":SENS:VOLT:DC:RANG {dmm_range}")
        self.dmm.interface.write(f":SENS:VOLT:DC:NPLC {nplc}")
        self.dmm.interface.write(":SENS:VOLT:DC:AZER ON")

        self.dmm.interface.write(":SENS:VOLT:DC:AVER:STAT OFF") # Disable averaging    

    def measure_voltage_avg(self, n_avg=10, delay=0.05):
        vals = []
        for _ in range(n_avg):
            v = self.dmm.measure_voltage_dc()
            vals.append(v)
            time.sleep(delay)
        mean = np.mean(vals)
        std = np.std(vals)
        return mean, std, vals
    
    def measure_voltage_dc(self):
        return self.dmm.interface.query(":MEAS:VOLT:DC?")

    def set_message(self, line1, line2=""):
        self.dmm.set_custom_message(line1, line2)

    def write(self, command: str):
        self.dmm.interface.write(command)

if __name__ == '__main__':
    IP = '172.16.1.51'
    dmm_ctrl = Dmm6500Controller(IP)

    dmm_ctrl.set_message("INL Test", "Running...")
    mean_v, std_v, all_v = dmm_ctrl.measure_voltage_avg(n_avg=16)
    print(f"Mean: {mean_v:.7f} V, Stddev: {std_v:.7g} V")
    dmm_ctrl.set_message("Test Done")
