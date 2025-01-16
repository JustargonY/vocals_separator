from frozendict import frozendict

SAMPLE_RATE = 44100
SIGNAL_NORMALIZATION_CONSTANT = 32767


STFT_DEFAULT_PARAMETERS = frozendict({
    "fs": SAMPLE_RATE,
    "nperseg": 1024,
    "noverlap": 512,
    "window": "hamming"
    })
