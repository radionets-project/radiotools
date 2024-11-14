import numpy as np
from numpy.testing import assert_array_equal, assert_raises


class TestPlottingUtils:
    def setup_class(self):
        self._HEADER = {
            "CDELT2": 2.7777778241006e-08,
            "CRPIX1": 1023.0,
            "CRPIX2": 1024.0,
            "NAXIS1": 2048,
            "NAXIS2": 2048,
        }

    def test_px2radec(self):
        from radiotools.plotting import px2radec

        _xlim = (0, 2048)
        _ylim = (0, 2048)
        _xticks = np.array([0, 256, 512, 768, 1024, 1280, 1536, 1792, 2048])
        _yticks = np.array([0, 256, 512, 768, 1024, 1280, 1536, 1792, 2048])
        _xticklabels = [
            "-102.40",
            "-76.80",
            "-51.20",
            "-25.60",
            "0.00",
            "25.60",
            "51.20",
            "76.80",
            "102.40",
        ]
        _yticklabels = [
            "-102.40",
            "-76.80",
            "-51.20",
            "-25.60",
            "0.00",
            "25.60",
            "51.20",
            "76.80",
            "102.40",
        ]

        (xlim, ylim, xticks, yticks, xticklabels, yticklabels) = px2radec(self._HEADER)

        assert xlim == _xlim
        assert ylim == _ylim
        assert_array_equal(xticks, _xticks)
        assert_array_equal(yticks, _yticks)
        assert_array_equal(xticklabels, _xticklabels)
        assert_array_equal(yticklabels, _yticklabels)

    def test_px2radec_xylim(self):
        from radiotools.plotting import px2radec

        _xlim = (922, 1122)
        _ylim = (923, 1123)
        _xticks = np.array([922, 947, 972, 997, 1022, 1047, 1072, 1097, 1122])
        _yticks = np.array([923, 948, 973, 998, 1023, 1048, 1073, 1098, 1123])
        _xticklabels = [
            "-10.00",
            "-7.50",
            "-5.00",
            "-2.50",
            "0.00",
            "2.50",
            "5.00",
            "7.50",
            "10.00",
        ]
        _yticklabels = [
            "-10.00",
            "-7.50",
            "-5.00",
            "-2.50",
            "0.00",
            "2.50",
            "5.00",
            "7.50",
            "10.00",
        ]

        (xlim, ylim, xticks, yticks, xticklabels, yticklabels) = px2radec(
            self._HEADER, xlim=(100, 100), ylim=(100, 100)
        )

        assert xlim == _xlim
        assert ylim == _ylim
        assert_array_equal(xticks, _xticks)
        assert_array_equal(yticks, _yticks)
        assert_array_equal(xticklabels, _xticklabels)
        assert_array_equal(yticklabels, _yticklabels)

    def test_px2radec_raise(self):
        from radiotools.plotting import px2radec

        assert_raises(ValueError, px2radec, self._HEADER, unit="kiloarcsecond")

    def test_px2radec_num_ticks(self):
        from radiotools.plotting import px2radec

        _xticklabels = [
            "-102.40",
            "-76.80",
            "-51.20",
            "-25.60",
            "0.00",
            "25.60",
            "51.20",
            "76.80",
            "102.40",
        ]
        _yticklabels = [
            "-102.40",
            "-76.80",
            "-51.20",
            "-25.60",
            "0.00",
            "25.60",
            "51.20",
            "76.80",
            "102.40",
        ]

        (_, _, _, _, xticklabels, yticklabels) = px2radec(
            self._HEADER, num_ticks=(9, 9)
        )

        assert_array_equal(xticklabels, _xticklabels)
        assert_array_equal(yticklabels, _yticklabels)

    def test_px2radec_axis(self):
        import matplotlib.pyplot as plt

        from radiotools.plotting import px2radec

        _xticklabels = [
            "-102.40",
            "-76.80",
            "-51.20",
            "-25.60",
            "0.00",
            "25.60",
            "51.20",
            "76.80",
            "102.40",
        ]
        _yticklabels = [
            "-102.40",
            "-76.80",
            "-51.20",
            "-25.60",
            "0.00",
            "25.60",
            "51.20",
            "76.80",
            "102.40",
        ]

        fig, ax = plt.subplots()

        px2radec(self._HEADER, ax=ax)

        assert_array_equal(
            [txt.get_text() for txt in ax.get_xticklabels()], _xticklabels
        )
        assert_array_equal(
            [txt.get_text() for txt in ax.get_yticklabels()], _yticklabels
        )
