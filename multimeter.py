import time
import numpy as np
from zolve_instruments import Dmm6500

class Dmm6500Controller:
    def __init__(self, ip_address, timeout_sec=5):
        self.dmm = Dmm6500(ip_address, timeout_sec)
        self.configure_for_precise_dc()

    def configure_for_precise_dc(self, nplc=5, autozero=True, dmm_range=10):
        self.dmm.interface.write(":SENS:FUNC 'VOLT:DC'")
        self.dmm.interface.write(f":SENS:VOLT:DC:RANG {dmm_range}")
        self.dmm.interface.write(f":SENS:VOLT:DC:NPLC {nplc}")
        if autozero:
            self.dmm.interface.write(":SENS:VOLT:DC:AZER ON")
        else:
            self.dmm.interface.write(":SENS:VOLT:DC:AZER OFF")


    def measure_voltage_avg(self, n_avg=10, delay=0.05):
        """Average n_avg voltage readings, with a delay between reads."""
        vals = []
        for _ in range(n_avg):
            v = self.dmm.measure_voltage_dc()
            vals.append(v)
            time.sleep(delay)
        mean = np.mean(vals)
        std = np.std(vals)
        return mean, std, vals

    def set_message(self, line1, line2=""):
        self.dmm.set_custom_message(line1, line2)

if __name__ == '__main__':
    IP = '172.16.1.51'
    dmm_ctrl = Dmm6500Controller(IP)

    dmm_ctrl.set_message("INL Test", "Running...")
    mean_v, std_v, all_v = dmm_ctrl.measure_voltage_avg(n_avg=16)
    print(f"Mean: {mean_v:.7f} V, Stddev: {std_v:.7g} V")
    dmm_ctrl.set_message("Test Done")
