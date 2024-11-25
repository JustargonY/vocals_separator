"""
This module handles loading data from the MUSDB18 and ccMixter datasets.
"""

import os
import tensorflow as tf
import numpy as np
import scipy.io.wavfile as wav

# the audio sample rate, common for both datasets
SAMPLE_RATE = 44100


class UnsupportedWavFileException(Exception):
    """
    An exception thrown when a wav file has a sample rate other than 44100Hz or if the file has
    more or less than 2 channels.
    """


def read_wav_file_to_array(file_path) -> np.ndarray:
    """
    Reads a wav file into a numpy array using the scipy.io.wavfile.read function.
    Asserts that the sample rate is correct and that there are 2 audio channels.
    Returns just the data array. 
    """
    file_path = file_path.numpy().decode("utf-8")
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
