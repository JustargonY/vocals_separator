import numpy as np
from .preprocessing import compute_stft
from .postprocessing import compute_inverse_stft


def load_model():
    """
    Loads prediction model from the file system into memory.

    :return: the model
    """
    # TODO


# should invoke load_model at the start


def predict(spectrogram):
    """
    Calls model.predict()

    :return:
    """
    # TODO


def predict_mocked(spectrogram):
    """
    Mocked version of _predict, returns random 0-1 vectors.

    :return:
    """
    random_guess = np.random.uniform(0, 1, spectrogram.shape[0] * spectrogram.shape[1]) > 0.5
    return random_guess.astype(np.float32)


def predict_signal(input_signal, **stft_params) -> tuple[np.ndarray, np.ndarray]:
    """
    Takes signal data from a wav file as input, returns a (vocal, instrumental) tuple
    of signal data, which can be saved to a wav file.

    :param input_signal: an array of shape (N_samples, 2) containing signal data for both channels
    :param stft_params: parameters to pass to stft and istft
    :return: tuple (vocal, instrumental), where both elements are 2D arrays of shape (N_samples, 2)
    """
    left = compute_stft(input_signal[:, 0], **stft_params)
    right = compute_stft(input_signal[:, 1], **stft_params)

    mask_left = predict(left)
    mask_right = predict(right)

    vocal = np.stack((
        compute_inverse_stft(np.where(mask_left, left, 0), **stft_params),
        compute_inverse_stft(np.where(mask_right, right, 0), **stft_params)
    ))
    instrumental = input_signal - vocal

    return vocal, instrumental
