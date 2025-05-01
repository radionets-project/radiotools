import numpy as np
from numpy.testing import assert_allclose

from radiotools.utils import rms


class TestRMS:
    def test_1d_array(self):
        """Test 1d array."""
        a = np.array([10, 0, 150, 40, 30, 20, -50, -60, 5])
        expected = np.sqrt(1 / 9 * 31625)

        assert_allclose(rms(a), expected)

    def test_2d_array_axis_none(self):
        """Test 2d array with axis=None."""
        a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        expected = 5.627314338711377

        assert_allclose(rms(a, axis=None), expected)

    def test_2d_array_axis_0(self):
        """Test 2d array with axis=0."""
        a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        expected = np.array(
            [
                4.69041575982343,
                5.5677643628300215,
                6.48074069840786,
            ]
        )

        assert_allclose(rms(a, axis=0), expected)

    def test_2d_array_axis_1(self):
        """Test 2d array with axis=1."""
        a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        expected = np.array(
            [
                2.160246899469287,
                5.066228051190222,
                8.04155872120988,
            ]
        )

        assert_allclose(rms(a, axis=1), expected)

    def test_input_ndim_0(self):
        """Test input with ndim=0"""
        a = np.int16(10)

        expected = 10.0

        assert rms(a) == expected
