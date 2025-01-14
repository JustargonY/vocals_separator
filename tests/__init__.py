import unittest
import os
import numpy as np
from vocalseparatordiploma import preprocessing


class FileLoadTest(unittest.TestCase):
    def test_file_loads(self):
        path = os.path.join(os.path.dirname(__file__), "test.wav")
        # should not throw exceptions
        signal = preprocessing.read_track(path)

        self.assertTrue(isinstance(signal, np.ndarray))


class PreprocessingTest(unittest.TestCase):
    def test_generate_windows(self):
        # some axample numbers
        spectrogram = np.array([
            [1, 2, 3, 4, 5],
            [3, 4, 5, 1, 2],
            [5, 4, 3, 2, 1]
        ])
        result = np.array(list(preprocessing.generate_windows(spectrogram, 3)))

        all_ok = (result == np.array([
            [
                [0, 1, 2],
                [0, 3, 4],
                [0, 5, 4]
            ],
            [
                [1, 2, 3],
                [3, 4, 5],
                [5, 4, 3]
            ],
            [
                [2, 3, 4],
                [4, 5, 1],
                [4, 3, 2]
            ],
            [
                [3, 4, 5],
                [5, 1, 2],
                [3, 2, 1]
            ],
            [
                [4, 5, 0],
                [1, 2, 0],
                [2, 1, 0]
            ]
        ])).all()
        self.assertTrue(all_ok)


if __name__ == "__main__":
    unittest.main()
