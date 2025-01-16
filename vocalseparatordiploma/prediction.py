import os
import numpy as np

from tensorflow.keras import Model
from .preprocessing import compute_stft, generate_windows
from .postprocessing import compute_inverse_stft
import tensorflow.keras.saving as saving


def load_model():
    """
    Loads prediction model from the file system into memory.

    :return: the model
    """

    path = os.path.join(os.path.dirname(__file__), "model.keras")
    model = Model()
    model.load_weights(filepath=path)
    return model


def predict(model, spectrogram):
    """
    Mocked version of _predict, returns random 0-1 vectors.

    :return:
    """

    return np.stack(
        [model.predict(frame) for frame in generate_windows(spectrogram)]
    )


def _get_inverse_mask(mask: np.array):
    """
    Reverse 0 and 1 in mask.
    Used to get instrumental mask from vocal mask.

    :return: 1 - mask
    """
    return 1 - mask


def predict(spectrogram: np.array):
    """
    Calls model.predict()

    :return:
    """

    spectrum_width, length = spectrogram.shape
    predicted_mask = np.zeros((spectrum_width, length))

    for batch_data, batch_indices in generate_windows(spectrogram):
        batch_predictions = model.predict(batch_data, verbose=0)
        for idx, pred_mask in zip(batch_indices, batch_predictions):
            predicted_mask[:, idx] = pred_mask

    return (predicted_mask >= 0.5).astype(np.float32)


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
        compute_inverse_stft(np.where(mask_right, right, 0), **stft_params),
    ), axis=1)
    instrumental = np.stack((
        compute_inverse_stft(np.where(mask_left_instr, left, 0), **stft_params),
        compute_inverse_stft(np.where(mask_right_instr, right, 0), **stft_params),
    ), axis=1)

    return vocal, instrumental
