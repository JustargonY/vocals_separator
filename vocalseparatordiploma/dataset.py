"""
This module handles loading data from the MUSDB18 and ccMixter datasets.
"""

import os
import tensorflow as tf
import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import stft

# the audio sample rate, common for both datasets
SAMPLE_RATE = 44100


def musdb18_get_file_names(dataset_path: str, train_test="train") -> tf.data.Dataset:
    """
    Create a tensorflow Dataset of filepaths of the Musdb18 Dataset.
    """
    if train_test not in ["train", "test"]:
        raise ValueError("Incorrect value for train_test: must be \"train\" or \"test\"")

    subdirectories = filter(os.path.isdir, os.listdir(dataset_path))
    return tf.data.Dataset.from_tensor_slices(
        list(map(lambda path: os.path.join(path, train_test, "mixture.wav"), subdirectories)),
        list(map(lambda path: os.path.join(path, train_test, "vocals.wav"), subdirectories))
    )


def ccmixter_get_file_names(dataset_path: str) -> tf.data.Dataset:
    """
    Create a tensorflow Dataset of filepaths of the ccMixter Dataset.
    """
    subdirectories = os.listdir(dataset_path)
    return tf.data.Dataset.from_tensor_slices(
        list(map(lambda path: os.path.join(path, "mix.wav"), subdirectories)),
        list(map(lambda path: os.path.join(path, "source-02.wav"), subdirectories))
    )


def read_wav_file_to_array(file_path) -> np.ndarray:
    """
    Reads a wav file into a numpy array using the scipy.io.wavfile.read function.
    Asserts that the sample rate is correct if assertions are enabled.
    Returns just the data array. 
    """
    file_path = file_path.numpy().decode("utf-8")
    # sr is the sample rate of the audio file
    sr, audio = wav.read(file_path)
    assert sr == SAMPLE_RATE

    return audio.astype(np.float32)


def transform_track_stft(signal, sample_rate=SAMPLE_RATE, nperseg=2048, noverlap=1024):
    """Transforms both channels of a stereo signal using STFT."""
    params = {"fs": sample_rate, "nperseg": nperseg, "noverlap": noverlap, "window": "hamming"}
    frequencies, times, transformed_left = stft(signal[:, 0], **params)
    frequencies, times, transformed_right = stft(signal[:, 1], **params)

    return frequencies, times, transformed_left, transformed_right


def get_windows_for_one_chanel(spectrogram, window_size=5):
    """A generator returning windows of the given width. The edges are padded with zeros."""
    spectrum_width, length = spectrogram.shape
    zeros_for_window = np.zeros((window_size // 2, spectrum_width))
    data_for_windows = np.concatenate([zeros_for_window, spectrogram.T, zeros_for_window])

    for i in range(length):
        yield data_for_windows[i:i+window_size]
