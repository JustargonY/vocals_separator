"""Are we going to need this???"""

import numpy as np
from scipy.signal import stft, istft
from .dataset import SAMPLE_RATE
from .preprocessing import generate_windows


def transform_track_stft(signal, sample_rate=SAMPLE_RATE, nperseg=2048, noverlap=1024):
    """Transforms both channels of a stereo signal using STFT."""
    params = {"fs": sample_rate, "nperseg": nperseg, "noverlap": noverlap, "window": "hamming"}
    frequencies, times, transformed_left = stft(signal[:, 0], **params)
    frequencies, times, transformed_right = stft(signal[:, 1], **params)

    return frequencies, times, transformed_left, transformed_right


def predict_for_window(window):
    """Returns a binary mask prediction for a single window."""
    # TODO load a pretrained model
    pass


def predict_binary_mask(spectrogram) -> np.ndarray:
    """Returns a binary mask to be applied to the spectrogram."""
    windows = generate_windows(spectrogram)
    results = np.zeros(spectrogram.shape)

    for idx, window in enumerate(windows):
        results[:, idx] = predict_for_window(window)

    return results


def predict(input_signal, **stft_params):
    """
    Takes signal data from a wav file as input, returns a (vocal, instrumental) tuple
    of signal data, which can be saved to a wav file.
    """
    freq, times, left, right = transform_track_stft(input_signal, **stft_params)

    mask_left = predict(left)
    mask_right = predict(right)

    vocal = np.stack((
        istft(np.where(mask_left, left, 0), **stft_params),
        istft(np.where(mask_right, right, 0), **stft_params)
    ))
    instrumental = input_signal - vocal

    return vocal, instrumental
