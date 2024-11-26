import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import stft, istft


def write_track(audio_path: str, sr: int, signal: np.array):
    """
    writes track into file
    """
    wav.write(audio_path, sr, signal.astype(np.int16))


def get_mocked_mask(zxx):
    """
    mocked function to generate mask (originally prediction output)
    :return: binary mask as np.array
    """
    return np.zeros(zxx.shape)


def combine_prediction_outputs():
    """
    connects all binary vectors from model output into one mask
    :return:
    """
    pass


def apply_mask(zxx: np.array, mask: np.array):
    """
    applies binary mask on mix
    :return: mixes as np.array
    """
    instr_mask = _get_inverse_mask(mask)
    vocals = zxx * mask
    instrumental = zxx * instr_mask
    return vocals, instrumental


def _get_inverse_mask(mask: np.array):
    """
    reverse 0 and 1 in binary mask
    used to get instrumental mask from vocal mask
    :return: inversed mask as np.array
    """
    return (mask == 0).astype(np.int32)


def compute_inverse_stft(sr: int, zxx_l: np.array, zxx_r: np.array):
    """
    computes inverse STFT of signal
    :return: istft of left and right channels as np.array
    """
    _, istft_l = istft(zxx_l, fs=sr, nperseg=1024, noverlap=512)
    _, istft_r = istft(zxx_r, fs=sr, nperseg=1024, noverlap=512)
    res = (np.stack((istft_l, istft_r), axis=-1) * 32768).astype(np.int16)
    return res
