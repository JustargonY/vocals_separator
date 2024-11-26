import numpy as np


def load_model():
    """
    Loads prediction model from the file system into memory.

    :return: the model
    """


# should invoke load_model at the start


def predict(spectrogram):
    """
    Calls model.predict()

    :return:
    """


def predict_mocked(spectrogram):
    """
    Mocked version of _predict, returns random 0-1 vectors.

    :return:
    """
    random_guess = np.random.uniform(0, 1, spectrogram.shape[0] * spectrogram.shape[1]) > 0.5
    return random_guess.astype(np.float32)
