from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from casatools.table import table
from scipy.constants import c


class Gridder:
    """
    A tool to grid and plot raw radio interferometric measurements

    """

    def __init__(self):
        None

    """
    Creates plots of the UV coverage of the measurement and plots its dirty image

    Parameters
    ----------
    uv_crop: tuple of array_like, optional
    The part of the UV coverage to display. Has to be a tuple of the bounds (x_bounds, y_bounds).

    uv_exp: float, optional
    The exponent of the UV coverage, e.g. for better visibility

    uv_norm: matplotlib.colors.norm, optional
    The norm to apply to the plot of the UV coverage, e.g. matplotlib.colors.LogNorm

    di_crop: tuple of array_like, optional
    The part of the dirty image to display. Has to be a tuple of the bounds (x_bounds, y_bounds).

    di_exp: float, optional
    The exponent of the dirty image, e.g. for better visibility

    uv_norm: matplotlib.colors.norm, optional
    The norm to apply to the plot of the dirty image, e.g. matplotlib.colors.LogNorm

    fig_size: array_like, optional
    The size of the plots

    """

    def plot(
        self,
        uv_crop=([None, None], [None, None]),
        uv_exp=1,
        uv_norm=None,
        di_crop=([None, None], [None, None]),
        di_exp=1,
        di_norm=None,
        figsize=[20, 10],
    ):
        plt.rcParams["figure.figsize"] = figsize

        fig, ax = plt.subplots(1, 2, layout="constrained")

        im1 = ax[0].imshow((self.mask**uv_exp), cmap="inferno", norm=uv_norm)
        ax[0].set_title("UV Plot")
        ax[0].set_xlabel("U")
        ax[0].set_ylabel("V")
        ax[0].set_xlim(left=uv_crop[0][0], right=uv_crop[0][1])
        ax[0].set_ylim(bottom=uv_crop[1][0], top=uv_crop[1][1])
        fig.colorbar(im1, ax=ax[0], shrink=0.8)

        im2 = ax[1].imshow(
            np.rot90(self.dirty_img**di_exp, 3), cmap="inferno", norm=di_norm
        )
        ax[1].set_xlim(left=di_crop[0][0], right=di_crop[0][1])
        ax[1].set_ylim(bottom=di_crop[1][0], top=di_crop[1][1])
        ax[1].set_title("Dirty image")
        fig.colorbar(im2, ax=ax[1], shrink=0.8, label="Jy/px")

        return fig, ax

    """
    Internal method to calculate the mask (UV coverage) and the dirty image

    Parameters
    ----------
    uu: array_like
    The U baseline coordinates in meters

    vv: array_like
    The U baseline coordinates in meters

    stokes_i: array_like
    The Stokes I parameters of the measurement

    """

    def _create_attributes(self, uu, vv, stokes_i):
        u = uu * self.freq / c
        v = vv * self.freq / c

        real = stokes_i.real.T
        imag = stokes_i.imag.T

        samps = np.array(
            [
                np.append(u.ravel(), -u.ravel()),
                np.append(v.ravel(), -v.ravel()),
                np.append(real.ravel(), real.ravel()),
                np.append(imag.ravel(), -imag.ravel()),
            ]
        )

        N = self.img_size

        delta_l = self.fov / N
        delta = (N * delta_l) ** (-1)

        bins = (
            np.arange(start=-(N / 2) * delta, stop=(N / 2 + 1) * delta, step=delta)
            - delta / 2
        )

        mask, *_ = np.histogram2d(samps[0], samps[1], bins=[bins, bins], density=False)
        mask[mask == 0] = 1

        mask_real, x_edges, y_edges = np.histogram2d(
            samps[0], samps[1], bins=[bins, bins], weights=samps[2], density=False
        )
        mask_imag, x_edges, y_edges = np.histogram2d(
            samps[0], samps[1], bins=[bins, bins], weights=samps[3], density=False
        )
        mask_real /= mask
        mask_imag /= mask

        self.mask = mask
        self.dirty_img = np.abs(
            np.rot90(
                np.fft.fftshift(
                    np.fft.ifft2(np.fft.fftshift(mask_real + 1j * mask_imag))
                ),
                3,
            )
        )[:, ::-1]
        return self

    """
    Initializes the Gridder with a measurement which is saved in a FITS file

    Parameters
    ----------
    fits_path: str
    The path of the FITS file

    img_size: int
    The pixel size of the image

    fov: float
    The field of view (pixel size * image size) of the image in arcseconds

    """

    @classmethod
    def from_fits(cls, fits_path, img_size, fov):
        path = Path(fits_path)

        if not path.is_file():
            raise FileNotFoundError(
                f"The file {path} could not be found! "
                f"You have to select a valid .fits file!"
            )

        cls = cls()
        cls.fits_path = fits_path
        cls.img_size = img_size
        cls.fov = fov * np.pi / (3600 * 180)

        file = fits.open(fits_path)

        data = file[0].data.T

        uu = data["UU--"].T * c
        vv = data["VV--"].T * c

        cls.freq = file[0].header["CRVAL4"]
        stokes_i = np.reshape(
            file[0].data["DATA"].T[:, 0:2][0, 0]
            + file[0].data["DATA"].T[:, 0:2][1, 0] * 1j
            + file[0].data["DATA"].T[:, 0:2][0, 1]
            + file[0].data["DATA"].T[:, 0:2][1, 1] * 1j,
            (file[0].data["DATA"].T.shape[6], 1),
        )

        return cls._create_attributes(uu, vv, stokes_i)

    """
    Initializes the Gridder with a measurement which is saved in a NRAO CASA measurement set

    Parameters
    ----------
    ms_path: str
    The path of the measurement set directory

    img_size: int
    The pixel size of the image

    fov: float
    The field of view (pixel size * image size) of the image in arcseconds

    """

    @classmethod
    def from_ms(cls, ms_path, img_size, fov):
        if ms_path[-1] != "/":
            ms_path += "/"

        path = Path(ms_path)

        if not path.is_dir():
            raise NotADirectoryError(
                f"This measurement set does not exist under the path {ms_path}"
            )

        cls = cls()
        cls.ms_path = ms_path
        cls.img_size = img_size
        cls.fov = fov * np.pi / (3600 * 180)

        data = table(ms_path).getcol("DATA").T

        uvw = table(ms_path).getcol("UVW").T

        cls.freq = table(ms_path + "SPECTRAL_WINDOW").getcol("CHAN_FREQ").T

        uvw = np.repeat(uvw[None], 1, axis=0)
        uu = uvw[:, :, 0]
        vv = uvw[:, :, 1]

        stokes_i = data[:, :, 0] + data[:, :, 1]

        return cls._create_attributes(uu, vv, stokes_i)
