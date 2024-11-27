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


def predict(spectrogram: np.array):
    """
    Calls model.predict()

    :return:
    """
    # TODO


def predict_mocked(spectrogram: np.array):
    """
    Mocked version of _predict, returns random 0-1 vectors.

    :return:
    """
    random_guess = np.random.uniform(0, 1, spectrogram.shape) > 0.5
    return random_guess.astype(np.float32)


def _get_inverse_mask(mask: np.array):
    """
    Reverse 0 and 1 in mask.
    Used to get instrumental mask from vocal mask.

    :return: 1 - mask
    """
    return 1 - mask


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

    # mask_left = predict(left)
    # mask_right = predict(right)

    mask_left = predict_mocked(left)
    mask_right = predict_mocked(right)

    mask_left_instr = _get_inverse_mask(mask_left)
    mask_right_instr = _get_inverse_mask(mask_right)

    vocal = np.stack((
        compute_inverse_stft(np.where(mask_left, left, 0), **stft_params),
        compute_inverse_stft(np.where(mask_right, right, 0), **stft_params),
    ), axis=1)
    instrumental = np.stack((
        compute_inverse_stft(np.where(mask_left_instr, left, 0), **stft_params),
        compute_inverse_stft(np.where(mask_right_instr, right, 0), **stft_params),
    ), axis=1)

    return vocal, instrumental
