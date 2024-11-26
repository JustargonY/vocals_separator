import scipy.io.wavfile as wav
import numpy as np
from scipy.signal import stft
from .dataset import SAMPLE_RATE


class UnsupportedWavFileException(Exception):
    """
    An exception thrown when a wav file has a sample rate other than 44100Hz or if the file has
    more or less than 2 channels.
    """


def read_track(file_path: str) -> np.ndarray:
    """
    Reads a wav file into a numpy array using the scipy.io.wavfile.read function.
    Asserts that the sample rate is correct and that there are 2 audio channels.
    Returns just the data array. 

    :param file_path: path to the wav file
    :returns: An array of shape (N_samples, 2) containing signal data of both audio channels
    """
    # sr is the sample rate of the audio file
    sample_rate, audio = wav.read(file_path)

    if sample_rate != SAMPLE_RATE:
        raise UnsupportedWavFileException(
            "Cannot load file: audio files for processing must have a sample rate of "
            f"{SAMPLE_RATE}. Sample rate of {file_path} is {sample_rate}."
            )
    elif audio.ndim < 2:
        raise UnsupportedWavFileException("Cannot load file: expected 2 audio channels, found 1.")
    elif audio.shape[1] != 2:
        raise UnsupportedWavFileException(
            f"Cannot load file: expected 2 audio channels, found {audio.shape[1]}.")

    return audio.astype(np.float32)



def compute_stft(signal, *args, **kwargs):
    """
    Applies STFT on the signal. A convienience function that calls scipy.signal.stft.

    :param signal: a 1D array containing signal data for a single channel
    :param args: parameters for scipy.signal.stft
    :param kwargs: parameters for scipy.signal.stft
    :return: The spectrogram
    """
    return stft(signal, *args, **kwargs)


def generate_windows(spectrogram, window_size=5):
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
    zeros_for_window = np.zeros((window_size // 2, spectrum_width))
    data_for_windows = np.concatenate([zeros_for_window, spectrogram.T, zeros_for_window])

    for i in range(length):
        yield data_for_windows[i:i+window_size]


def get_ideal_binary_mask():
    """
    generates IBM from mix and vocals
    :return:
    """
    # ?????????????? what is this supposed to be doing
