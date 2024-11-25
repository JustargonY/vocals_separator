"""This submodule exposes an API for the trained model."""

import numpy as np
from scipy.signal import istft
from .dataset import transform_track_stft, get_windows_for_one_chanel

def predict_for_window(window):
    """Returns a binary mask prediction for a single window."""
    # TODO load a pretrained model
    pass


def predict_binary_mask(spectrogram) -> np.ndarray:
    """Returns a binary mask to be applied to the spectrogram."""
    windows = get_windows_for_one_chanel(spectrogram)
    results = np.zeros(spectrogram.shape)

    for idx, window in enumerate(windows):
        results[:, idx] = predict_for_window(window)

    return results


def predict(input_signal, **stft_params):
    """
    Takes signal data from a wav file as input, returns a (vocal, instrumental) tuple
    of signal data, which can be saved to a wav file
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
