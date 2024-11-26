"""
This module handles loading data from the MUSDB18 and ccMixter datasets.
"""

import os
import tensorflow as tf

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
