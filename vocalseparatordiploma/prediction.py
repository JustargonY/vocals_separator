import numpy as np
from .preprocessing import compute_stft, generate_windows
from .postprocessing import compute_inverse_stft
import tensorflow.keras.saving as saving


def load_prediction_model():
    """
    Loads prediction model from the file system into memory.

    :return: the model
    """

    def extract_central_frame(x):
        return x[:, 15, :]

    # path should be changed
    model = saving.load_model('C:/diploma/models/cnn_rnn_super.keras',
                              custom_objects={"extract_central_frame": extract_central_frame})

    return model


# should invoke load_model at the start
model = load_prediction_model()


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

    mask_left = predict(np.abs(left))
    mask_right = predict(np.abs(right))

    # mask_left = predict_mocked(left)
    # mask_right = predict_mocked(right)

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
