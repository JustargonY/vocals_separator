import scipy.io.wavfile as wav
import numpy as np
from scipy.signal import stft
from .constants import SAMPLE_RATE, STFT_DEFAULT_PARAMETERS, SIGNAL_NORMALIZATION_CONSTANT


class UnsupportedWavFileException(Exception):
    """
    An exception thrown when a wav file has a sample rate other than 44100Hz or if the file has
    more or less than 2 channels.
    """


def read_track(file_path: str) -> np.ndarray:
    """
    Reads a wav file into a numpy array using the scipy.io.wavfile.read function.
    Asserts that the sample rate is correct and that there are 2 audio channels.
    Normalizes the signal by dividing it by the `SIGNAL_NORMALIZATION_CONSTANT`.
    Returns just the data array.

    :param file_path: path to the wav file
    :returns: An array of shape (N_samples, 2) containing signal data of both audio channels
    """
    sample_rate, audio = wav.read(file_path)

    # exceptions in case the file format is different than expected
    if sample_rate != SAMPLE_RATE:
        raise UnsupportedWavFileException(
            "Cannot load file: audio files for processing must have a sample rate of "
            f"{SAMPLE_RATE}. Sample rate of {file_path} is {sample_rate}."
            )
    if audio.ndim < 2:
        raise UnsupportedWavFileException("Cannot load file: expected 2 audio channels, found 1.")
    if audio.shape[1] != 2:
        raise UnsupportedWavFileException(
            f"Cannot load file: expected 2 audio channels, found {audio.shape[1]}.")

    return audio.astype(np.float32) / SIGNAL_NORMALIZATION_CONSTANT



def compute_stft(signal, **kwargs):
    """
    Applies STFT on the signal. A convienience function that calls scipy.signal.stft.
    For default parameters see `STFT_DEFAULT_PARAMETERS` in module constants.

    :param signal: a 1D array containing signal data for a single channel
    :param kwargs: parameters for `scipy.signal.stft`
    :return: The spectrogram
    """
    kwargs = STFT_DEFAULT_PARAMETERS | kwargs
    _, _, transformed = stft(signal, **kwargs)
    return transformed


def generate_windows(spectrogram, window_size=25):
    """
    A generator returning windows of the given width.
    For output number i the i-th frame is in the center of the window.
    The edges are padded with zeros.

    :param spectrogram: a 2D array of shape (N_frequencies, N_samples), the spectrogram of a single
                            audio channel
    :param window_size: the width of windows yielded by the generator
    :return: a generator object yielding arrays of shape (N_frequencies, window_size)
    """
    spectrum_width, length = spectrogram.shape
    zeros_for_window = np.zeros((spectrum_width, window_size // 2))
    data_for_windows = np.concatenate([zeros_for_window, spectrogram, zeros_for_window], 1)

    for i in range(length):
        yield data_for_windows[:, i:i+window_size]


def get_ideal_binary_mask(mixture_spectrogram, vocals_spectrogram, cutoff=0.8):
    """
    Creates a binary mask, where 1 means vocals are dominant, and 0 means the accompaniment
    is dominant. More specifically, the binary mask has a 1, if vocals > (mix * cutoff), where
    0.5 <= cutoff < 1.

    :param mixture_spectrogram: obtained from the mix through STFT
    :param vocals_spectrogram: obtained from the true vocals-only recording
    :return: A binary mask - an array containing only 1s and 0s of the same shape as
                    mixture_spectrogram
    """
    return np.where(vocals_spectrogram > mixture_spectrogram * cutoff, 1, 0)
