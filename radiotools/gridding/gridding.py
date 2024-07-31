import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.constants import c
from astropy.io import fits
from casatools.table import table
from matplotlib.colors import LogNorm, PowerNorm


class Gridder:
    """
    A tool to grid and plot raw radio interferometric measurements
    """

    def __init__(self):
        None

    def plot(
        self,
        uv_args={},
        mask_args={},
        mask_abs_args={},
        mask_phase_args={},
        dirty_img_args={},
        save_to=None,
        save_args={},
        figsize=[20, 20],
    ):
        """
        Creates a summary plot containing the gridded and ungridded uv coverage,
        the amplitude and phase of the visibilities and the dirty image

        Parameters
        ----------
        uv_args : dict, optional
            The function arguments for the uv coverage plot (arguments for the plot_ungridded_uv function)

        mask_args : dict, optional
            The function arguments for the uv mask plot (arguments for the plot_mask function)

        mask_abs_args : dict, optional
            The function arguments for the uv coverage plot (arguments for the plot_mask_absolute function)

        mask_phase_args : dict, optional
            The function arguments for the uv coverage plot (arguments for the plot_mask_phase function)

        dirty_img_args : dict, optional
            The function arguments for the uv coverage plot (arguments for the plot_dirty_image function)

        save_to : str, optional
            Path to save the figure to

        save_args : str, optional
            The arguments for the savefig function

        figsize : array_like, optional
            The size of the plots

        """

        fig, ax = plt.subplots(ncols=2, nrows=3, layout="constrained", figsize=figsize)
        ax = np.ravel(ax)

        titles = [
            "Ungerasterte $(u,v)$-Abdeckung",
            "Gerasterte $(u,v)$-Abdeckung",
            "Amplitude der Visibilities",
            "Phase der Visibilities",
            "Dirty Image",
        ]

        for i in range(0, 5):
            ax[i].set_title(titles[i])

        self.plot_ungridded_uv(**uv_args, fig=fig, ax=ax[0])
        self.plot_mask(**mask_args, fig=fig, ax=ax[1])
        self.plot_mask_absolute(**mask_abs_args, fig=fig, ax=ax[2])
        self.plot_mask_phase(**mask_phase_args, fig=fig, ax=ax[3])
        self.plot_dirty_image(**dirty_img_args, fig=fig, ax=ax[4])
        ax[5].set_axis_off()

        if save_to is not None:
            fig.savefig(save_to, **save_args)

    def plot_ungridded_uv(
        self,
        plot_args={"color": "royalblue", "s": 0.01},
        save_to=None,
        save_args={},
        fig=None,
        ax=None,
    ):
        """
        Plots the ungridded uv coverage

        Parameters
        ----------
        plot_args : dict, optional
            The arguments for the pyplot scatter plot of the uv tuples

        save_to : str, optional
            Path to save the figure to

        save_args : str, optional
            The arguments for the savefig function

        fig : matplotlib.figure.Figure, optional
            A figure to put the plot into

        ax : matplotlib.axes._axes.Axes, optional
            A axis to put the plot into

        """

        if ax is None:
            fig, ax = plt.subplots()

        ax.scatter(
            x=np.append(self.uu, -self.uu), y=np.append(self.vv, -self.vv), **plot_args
        )
        ax.set_xlabel("$u$ / $\\lambda$")
        ax.set_ylabel("$v$ / $\\lambda$")

        if save_to is not None:
            fig.savefig(save_to, **save_args)

        return fig, ax

    def plot_mask(
        self,
        crop=([None, None], [None, None]),
        plot_args={"cmap": "inferno", "norm": LogNorm()},
        colorbar_shrink=0.7,
        save_to=None,
        save_args={},
        fig=None,
        ax=None,
    ):
        """
        Plots the gridded uv coverage mask

        Parameters
        ----------
        crop : tuple of arrays, optional
            The cutout of the plot to display (e.g. ([-10, 10], [-15, 15])

        plot_args : dict, optional
            The arguments for the pyplot scatter plot of the uv tuples

        colorbar_shrink : float, optional
            The shrink parameter for the colorbar

        save_to : str, optional
            Path to save the figure to

        save_args : str, optional
            The arguments for the savefig function

        fig : matplotlib.figure.Figure, optional
            A figure to put the plot into

        ax : matplotlib.axes._axes.Axes, optional
            A axis to put the plot into

        """

        if None in (fig, ax) and not all(x is None for x in (fig, ax)):
            raise KeyError(
                "The parameters ax and fig have to be both None or not None!"
            )

        if ax is None:
            fig, ax = plt.subplots(layout="constrained")

        im0 = ax.imshow(self.mask, **plot_args)
        ax.set_xlim(crop[0][0], crop[0][1])
        ax.set_ylim(crop[1][0], crop[1][1])
        ax.set_xlabel("Pixel")
        ax.set_ylabel("Pixel")
        fig.colorbar(
            im0, ax=ax, shrink=colorbar_shrink, label="$(u,v)$ pro Pixel in 1 / px"
        )

        if save_to is not None:
            fig.savefig(save_to, **save_args)

        return fig, ax

    def plot_mask_absolute(
        self,
        crop=([None, None], [None, None]),
        plot_args={"cmap": "inferno", "norm": LogNorm()},
        colorbar_shrink=0.9,
        save_to=None,
        save_args={},
        fig=None,
        ax=None,
    ):
        """
        Plots the amplitude of the complex visibilities

        Parameters
        ----------
        crop : tuple of arrays, optional
            The cutout of the plot to display (e.g. ([-10, 10], [-15, 15])

        plot_args : dict, optional
            The arguments for the pyplot scatter plot of the uv tuples

        colorbar_shrink : float, optional
            The shrink parameter for the colorbar

        save_to : str, optional
            Path to save the figure to

        save_args : str, optional
            The arguments for the savefig function

        fig : matplotlib.figure.Figure, optional
            A figure to put the plot into

        ax : matplotlib.axes._axes.Axes, optional
            A axis to put the plot into

        """

        if None in (fig, ax) and not all(x is None for x in (fig, ax)):
            raise KeyError(
                "The parameters ax and fig have to be both None or not None!"
            )

        if ax is None:
            fig, ax = plt.subplots(layout="constrained")

        im = ax.imshow(np.absolute(self.mask_real + self.mask_imag * 1j), **plot_args)
        ax.set_xlim(crop[0][0], crop[0][1])
        ax.set_ylim(crop[1][0], crop[1][1])
        ax.set_xlabel("Pixel")
        ax.set_ylabel("Pixel")

        fig.colorbar(im, ax=ax, shrink=colorbar_shrink, label="Intensität in a.u.")

        if save_to is not None:
            fig.savefig(save_to, **save_args)

        return fig, ax

    def plot_mask_phase(
        self,
        crop=([None, None], [None, None]),
        plot_args={"cmap": "coolwarm"},
        colorbar_shrink=0.9,
        save_to=None,
        save_args={},
        fig=None,
        ax=None,
    ):
        """
        Plots the phase of the complex visibilities

        Parameters
        ----------
        crop : tuple of arrays, optional
            The cutout of the plot to display (e.g. ([-10, 10], [-15, 15])

        plot_args : dict, optional
            The arguments for the pyplot scatter plot of the uv tuples

        colorbar_shrink : float, optional
            The shrink parameter for the colorbar

        save_to : str, optional
            Path to save the figure to

        save_args : str, optional
            The arguments for the savefig function

        fig : matplotlib.figure.Figure, optional
            A figure to put the plot into

        ax : matplotlib.axes._axes.Axes, optional
            A axis to put the plot into

        """

        if None in (fig, ax) and not all(x is None for x in (fig, ax)):
            raise KeyError(
                "The parameters ax and fig have to be both None or not None!"
            )

        if ax is None:
            fig, ax = plt.subplots(layout="constrained")

        im = ax.imshow(np.angle(self.mask_real + self.mask_imag * 1j), **plot_args)
        ax.set_xlim(crop[0][0], crop[0][1])
        ax.set_ylim(crop[1][0], crop[1][1])
        ax.set_xlabel("Pixel")
        ax.set_ylabel("Pixel")

        fig.colorbar(im, ax=ax, shrink=colorbar_shrink, label="Phasendifferenz in rad")

        if save_to is not None:
            fig.savefig(save_to, **save_args)

        return fig, ax

    def plot_dirty_image(
        self,
        mode="real",
        crop=([None, None], [None, None]),
        exp=1,
        plot_args={"cmap": "inferno", "origin": "lower"},
        colorbar_shrink=1,
        save_to=None,
        save_args={},
        fig=None,
        ax=None,
    ):
        """
        Plots the dirty image generated by the iFFT of the visibilites

        Parameters
        ----------
        mode : str, optional
            The component or variant of the diry image to show
            (available: real, imag, abs)

        crop : tuple of arrays, optional
            The cutout of the plot to display (e.g. ([-10, 10], [-15, 15])

        exp : float, optional
            The exponent for the power norm to apply to the plot

        plot_args : dict, optional
            The arguments for the pyplot scatter plot of the uv tuples

        colorbar_shrink : float, optional
            The shrink parameter for the colorbar

        save_to : str, optional
            Path to save the figure to

        save_args : str, optional
            The arguments for the savefig function

        fig : matplotlib.figure.Figure, optional
            A figure to put the plot into

        ax : matplotlib.axes._axes.Axes, optional
            A axis to put the plot into

        """

        if None in (fig, ax) and not all(x is None for x in (fig, ax)):
            raise KeyError(
                "The parameters ax and fig have to be both None or not None!"
            )

        if ax is None:
            fig, ax = plt.subplots(layout="constrained")

        match mode:
            case "real":
                dirty_image = self.dirty_img
            case "imag":
                dirty_image = np.imag(self.dirty_img_cmplx)[:, ::-1]
            case "abs":
                dirty_image = np.absolute(self.dirty_img_cmplx)[:, ::-1]
            case _:
                dirty_image = self.dirty_img
                warnings.warn(
                    f"The mode {mode} does not exist. Use real, imag or abs. Using real by default"
                )

        dirty_image[dirty_image < 0] = 0

        norm = None if exp == 1 else PowerNorm(gamma=exp)

        im = ax.imshow(dirty_image, norm=norm, **plot_args)
        ax.set_xlabel("Pixel")
        ax.set_ylabel("Pixel")
        fig.colorbar(
            im,
            ax=ax,
            shrink=colorbar_shrink,
            label="$\\text{Flussdichte}\\text{ in }\\text{Jy/px}$",
        )

        if save_to is not None:
            fig.savefig(save_to, **save_args)

        return fig, ax

    def _create_attributes(self, uu, vv, stokes_i):
        """
        Internal method to calculate the mask (UV coverage) and the dirty image

        Parameters
        ----------
        uu : array_like
            The U baseline coordinates in units of wavelength

        vv : array_like
            The U baseline coordinates in units of wavelength

        stokes_i : array_like
            The Stokes I parameters of the measurement

        """

        u = uu * self.freq / c
        v = vv * self.freq / c

        self.uu = uu
        self.vv = vv

        self.u = u
        self.v = v

        stokes_i = stokes_i[:, 0]

        self.stokes_i = stokes_i

        real = stokes_i.real.T
        imag = stokes_i.imag.T

        samps = np.array(
            [
                np.append(-u.ravel(), u.ravel()),
                np.append(-v.ravel(), v.ravel()),
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
        self.mask_real = mask_real
        self.mask_imag = mask_imag
        self.dirty_img_cmplx = np.fft.fftshift(
            np.fft.ifft2(np.fft.fftshift(mask_real + 1j * mask_imag))
        )
        self.dirty_img = np.real(self.dirty_img_cmplx)[:, ::-1]

        return self

    @classmethod
    def from_fits(cls, fits_path, img_size, fov):
        """
        Initializes the Gridder with a measurement which is saved in a FITS file

        Parameters
        ----------
        fits_path : str
            The path of the FITS file

        img_size : int
            The pixel size of the image

        fov : float
            The field of view (pixel size * image size) of the image in arcseconds

        """

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

    @classmethod
    def from_ms(cls, ms_path, img_size, fov):
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
