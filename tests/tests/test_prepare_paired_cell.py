import unittest

from benchmarks.prepare_paired_cell import verify_matrix


class VerifySavedMatrixTests(unittest.TestCase):
    def test_accepts_cross_backend_roundoff(self):
        difference = verify_matrix(
            [["V1", "1", "2"], ["V2", "3", "4"]],
            {"dim": [2, 2], "data": [1 + 1e-12, 2, 3, 4]}, "Z",
        )
        self.assertLess(difference, 1e-10)

    def test_rejects_reordered_samples(self):
        with self.assertRaisesRegex(ValueError, "values/order"):
            verify_matrix(
                [["V2", "3", "4"], ["V1", "1", "2"]],
                {"dim": [2, 2], "data": [1, 2, 3, 4]}, "Z",
            )

    def test_rejects_shape_mismatch(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            verify_matrix([["V1", "1", "2"]],
                          {"dim": [2, 1], "data": [1, 2]}, "Z")
