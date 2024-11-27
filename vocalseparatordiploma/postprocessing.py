import os
import scipy.io.wavfile as wav
import numpy as np
from scipy.signal import istft
from .constants import SAMPLE_RATE, STFT_DEFAULT_PARAMETERS, SIGNAL_NORMALIZATION_CONSTANT


def write_track(data, filepath=None) -> None:
    """
    Writes track into file. A convenience function that calls `scipy.io.wavfile.write` with
    the sample rate of 44100Hz (equal to the sample rate of all tracks used in model training).
    Multiplies the signal data by SIGNAL_NORMALIZATION_CONSTANT

    :param data: an array of shape (N_samples, N_channels); N_channels should in our case
                    be generally equal to 2, but no checks are made
    :param filepath: string or open file handle; if None defaults to "output.wav" in the current
                    working directory
    """

    if filepath is None:
        filepath = os.path.join(os.getcwd(), "output.wav")

    wav.write(filepath, SAMPLE_RATE, (data * SIGNAL_NORMALIZATION_CONSTANT).astype(np.int16))


def combine_prediction_outputs(outputs):
    """
    Connects all binary vectors from model output into one mask

    :return: np.stack(outputs)
    """
    return np.stack(outputs)


def apply_mask(mixture_spectrogram, binary_mask):
    """
    Applies binary mask on the mix.

    :return: a (vocal,  instrumental) tuple of 2D arrays (spectrograms)
    """
    vocals = mixture_spectrogram * binary_mask
    return vocals, mixture_spectrogram - vocals


def compute_inverse_stft(spectrogram, **kwargs):
    """
    Computes inverse STFT of a 2D array, returning the original signal.
    A convienience function, calls `scipy.signal.istft`.
    The defaults are in `STFT_DEFAULT_PARAMETERS`, a dictionary in `constants.py`

    :param spectrogram: the 2D array to transform
    :param kwargs: other parameters used by scipy.signal.istft
    :return:
    """
    kwargs = STFT_DEFAULT_PARAMETERS | kwargs
    _, signal = istft(spectrogram, **kwargs)
    return signal
