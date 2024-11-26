import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import stft


def read_track(audio_path: str):
    """
    reads track from file
    :return: sample rate as int and audio as np.array
    """
    sr, audio = wav.read(audio_path)
    if audio.dtype == np.int16:
        audio = audio / 32768.0
    return sr, audio


def compute_stft(audio: np.array, sr: int):
    """
    applies STFT on signal
    :return: frequencies, times and magnitudes of left and right tracks as np.array
    """
    frequencies, times, Zxx_l = stft(audio[:, 0], fs=sr, nperseg=1024, noverlap=512)
    _, _, Zxx_r = stft(audio[:, 1], fs=sr, nperseg=1024, noverlap=512)

    return frequencies, times, Zxx_l, Zxx_r


def generate_windows():
    """
    generates input data for model
    :return:
    """
    pass
