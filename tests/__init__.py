import unittest
import os
import numpy as np
from vocalseparatordiploma import preprocessing, prediction, postprocessing


class FileLoadTest(unittest.TestCase):
    def test_file_loads(self):
        path = os.path.join(os.path.dirname(__file__), "test.wav")
        # should not throw exceptions
        signal = preprocessing.read_track(path)

        self.assertIsInstance(signal, np.ndarray)


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


class ModelTest(unittest.TestCase):
    def test_model_loads(self):
        path = os.path.join(os.path.dirname(__file__), "test.wav")
        signal = preprocessing.read_track(path)
        left = preprocessing.compute_stft(signal[:, 0])
        right = preprocessing.compute_stft(signal[:, 1])

        # try to load the model
        model = prediction.load_model()
        self.assertIsNotNone(model)

        pred_left = prediction.predict(model, left)
        pred_right = prediction.predict(model, right)

        # check if the output shape is ok
        self.assertEqual(left.shape, pred_left.shape)
        self.assertEqual(right.shape, pred_right.shape)

        # make sure the output format is ok
        self.assertTrue((pred_left <= 1).all())
        self.assertTrue((pred_left >= 0).all())
        self.assertTrue((pred_right <= 1).all())
        self.assertTrue((pred_right >= 0).all())


if __name__ == "__main__":
    unittest.main()
