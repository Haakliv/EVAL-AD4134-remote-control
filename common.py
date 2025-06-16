SLAVE_ODR_MAP = {
    0: 1_449_275.0,      # 1.449275 MHz
    1: 1_250_000.0,      # 1.25 MHz
    2: 1_123_596.0,      # 1.123596 MHz
    3: 800_000.0,        # 800 kHz
    4: 751_880.0,        # 751.88 kHz
    5: 645_161.0,        # 645.161 kHz
    6: 600_000.0,        # 600 kHz
    7: 363_636.0,        # 363.636 kHz
    8: 320_513.0,        # 320.513 kHz
    9: 250_000.0,        # 250 kHz
    10: 100_000.0,       # 100 kHz
    11: 50_000.0,        # 50 kHz
    12: 10_000.0,        # 10 kHz
    13: 1_000.0,         # 1 kHz
}

SINC_FILTER_MAP = {
    0: 'Sinc3',
    1: 'Sinc3 50Hz & 60Hz Rejection',
    2: 'Sinc6',
    3: 'Wideband01',
    4: 'Wideband04',
}

ADC_RES_BITS    = 24
MAX_INPUT_RANGE = 4.096
ADC_SCALE = MAX_INPUT_RANGE / (2 ** (ADC_RES_BITS - 1))
ADC_LSB = MAX_INPUT_RANGE / (2 ** (ADC_RES_BITS - 1))
DEFAULT_STEP_VPP = MAX_INPUT_RANGE * 0.9

KILO = 1e3
MICRO = 1e6