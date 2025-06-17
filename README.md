# EVAL-AD4134-remote-control

Automation scripts for characterising the **Analog Devices AD4134** ADC via the **EVAL-AD4134** board and **EVAL-SDP-CH1Z** controller using the Analog Devices ACE software. Developed for the TFE4580 Project Thesis at NTNU, spring 2025.
The toolkit utilizes:

* **Siglent SDG6022X** function generator  
* **Keithley DMM6500** 6 1/2-digit multimeter  
* **Agilent B2912A** precision SMU  

The SDG6022X and DMM6500 instruments are accessed over LAN/USB using the *zolve-instruments* PyVISA wrappers. The B2912A is controlled using a custom PyVISA wrapper.

The logs and results presented in the thesis are stored in subfolders within the **`logs/`** directory.

---

## Repository layout

| File / module        | Purpose                                                                                                                                                  |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`ace_client.py`**  | Thin wrapper for remote control of Analog Devices ACE.                                                                                                   |
| **`acgain.py`**      | Stand-alone script to estimate AC gain of the AFE                                                                                                        |
| **`acquisition.py`** | Capture and decode raw samples via `ace_client`                                                                                                          |
| **`cli.py`**         | Command-line front-end. Implements test routines: noise, DC gain/offset, INL, frequency response, dynamic performance (SFDR / THD / ENOB), settling time |
| **`common.py`**      | Project-wide constants and utility functions                                                                                                             |
| **`generator.py`**   | Control of the Siglent SDG6022X waveform generator                                                                                                       |
| **`multimeter.py`**  | Control of the Keithley DMM6500                                                                                                                          |
| **`source.py`**      | Control of the Agilent B2912A SMU                                                                                                                        |
| **`plotting.py`**    | Common plotting helpers                                                                                                                                  |
| **`processing.py`**  | Numeric post-processing of captured data                                                                                                                 |
| **`logs/`**          | Test logs (one sub-folder per run)                                                                                                                       |

---

## Installation

```bash
pip install -r requirements.txt   # includes zolve-instruments and PyVISA-py
