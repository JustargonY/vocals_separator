import os
import scipy.io.wavfile as wav
from .dataset import SAMPLE_RATE


def write_track(data, filepath=None):
    """
    Writes track into file. A convenience function that calls scipy.io.wavfile.write with
    the sample rate of 44100Hz (equal to the sample rate of all tracks used in model training).

    :param data: a numpy array of shape (N_samples, N_channels); N_channels should in our case
                    be generally equal to 2, but no checks are made
    :param filepath: string or open file handle; if None defaults to "output.wav" in the current working
                    directory
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


def _get_inverse_mask():
    """
    reverse 0 and 1 in mask
    used to get instrumental mask from vocal mask
    :return:
    """


def compute_inverse_stft():
    """
    computes inverse STFT of signal
    :return:
    """
