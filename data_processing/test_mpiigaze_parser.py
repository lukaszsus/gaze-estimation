from unittest import TestCase

from scripts.create_dataset.create_dataset_mpiigaze_processed_both_rgb import parse_mpiigaze


class TestMpiigazeParser(TestCase):
    def test_parse_mpiigaze(self):
        right_eye, left_eye = parse_mpiigaze(0, 1, 1)
        self.assertAlmostEqual(right_eye["gaze_theta"], -0.090232, 4)
        self.assertAlmostEqual(right_eye["gaze_phi"], 0.12283, 4)
        self.assertAlmostEqual(right_eye["headpose_theta"], 0.15266, 4)
        self.assertAlmostEqual(right_eye["headpose_phi"], 0.24220, 4)

