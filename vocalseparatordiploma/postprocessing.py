import os
import scipy.io.wavfile as wav
from scipy.signal import stft, istft
from .dataset import SAMPLE_RATE


def write_track(data, filepath=None) -> None:
    """
    Writes track into file. A convenience function that calls scipy.io.wavfile.write with
    the sample rate of 44100Hz (equal to the sample rate of all tracks used in model training).

    :param data: an array of shape (N_samples, N_channels); N_channels should in our case
                    be generally equal to 2, but no checks are made
    :param filepath: string or open file handle; if None defaults to "output.wav" in the current
                    working directory
    """

    if filepath is None:
        filepath = os.path.join(os.getcwd(), "output.wav")

    wav.write(filepath, SAMPLE_RATE, data)


def combine_prediction_outputs():
    """
    connects all binary vectors from model output into one mask
    :return:
    """

def apply_mask():
    """
    applies binary mask on mix
    :return:
    """


def _get_inverse_mask(mask):
    """
    reverse 0 and 1 in mask
    used to get instrumental mask from vocal mask
    :return: 1 - mask
    """
    return 1 - mask


def compute_inverse_stft(spectrogram, *args, **kwargs):
    """
    Computes inverse STFT of a 2D array, returning the original signal.
    A convienience function, calls scipy.signal.istft.

    :param spectrogram: the 2D array to transform
    :param args: other parameters used by scipy.signal.istft
    :param kwargs: other parameters used by scipy.signal.istft
    :return:
    """
    return istft(spectrogram, *args, **kwargs)
