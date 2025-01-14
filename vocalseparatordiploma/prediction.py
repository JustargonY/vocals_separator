import os
import numpy as np
from tensorflow.keras import Model
from .preprocessing import compute_stft, generate_windows
from .postprocessing import compute_inverse_stft


def load_model():
    """
    Loads prediction model from the file system into memory.

    :return: the model
    """
    path = os.path.join(os.path.dirname(__file__), "model.keras")
    model = Model.load_weights(path)
    return model


def predict(model, spectrogram):
    """
    Calls model.predict()

    :return:
    """
    return np.stack(
        [model.predict(frame) for frame in generate_windows(spectrogram)]
    )


def predict_mocked(spectrogram):
    """
    Mocked version of _predict, returns random 0-1 vectors.

    :return:
    """
    random_guess = np.random.uniform(0, 1, spectrogram.shape[0] * spectrogram.shape[1]) > 0.5
    return random_guess.astype(np.float32)


def predict_signal(input_signal: np.ndarray, **stft_params) -> tuple[np.ndarray, np.ndarray]:
    """
    Takes signal data from a wav file as input, returns a (vocal, instrumental) tuple
    of signal data, which can be saved to a wav file.

    :param input_signal: an array of shape (N_samples, 2) containing signal data for both channels
    :param stft_params: parameters to pass to stft and istft
    :return: tuple (vocal, instrumental), where both elements are 2D arrays of shape (N_samples, 2)
    """
    left = compute_stft(input_signal[:, 0], **stft_params)
    right = compute_stft(input_signal[:, 1], **stft_params)

    model = load_model()

    mask_left = predict(model, left)
    mask_right = predict(model, right)

    vocal = np.stack((
        compute_inverse_stft(np.where(mask_left, left, 0), **stft_params),
        compute_inverse_stft(np.where(mask_right, right, 0), **stft_params)
    ))
    instrumental = input_signal - vocal

    return vocal, instrumental
