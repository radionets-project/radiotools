import warnings
from pathlib import Path

import astropy.units as units
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.constants import c
from astropy.io import fits
from astropy.io.fits import PrimaryHDU
from matplotlib.patches import Ellipse
from matplotlib.ticker import ScalarFormatter
from radio_stats.cuts.dbscan_clean import dbscan_clean
from radio_stats.cuts.dyn_range import rms_cut

from radiotools.utils import beam2pix, pix2beam


class Fiducial:
    def __init__(self, fits_path: str):
        """Initialize a new fiducial image from a FITS file.

        Parameters
        ----------

        path : str
            The path to the FITS file of the fiducial image.

        """

        self._fits_path = Path(fits_path).resolve()

    def get_hdu(self):
        """Get the Primary Header Data Unit (HDU) of the FITS file.

        Returns
        -------

        astropy.io.fits.PrimaryHDU
            The HDU of the FITS file

        """
        return fits.open(self._fits_path)[0]

    def get_header(self, hdu: PrimaryHDU | None = None):
        """Get the FITS Header of the fiducial image.

        Parameters
        ----------

        hdu : astropy.io.fits.PrimaryHDU | None, optional
            The Primary Header Data Unit of the FITS file. Can be provided
            to avoid consecutive opening and closing of the file.
            If set to ``None``, the file will be opened.
            Default is ``None``.

        Returns
        -------

        astropy.io.fits.Header
            The FITS Header of the fiducial image
        """
        hdu = hdu if hdu is not None else self.get_hdu()

        return hdu.header

    def get_metadata(self, hdu: PrimaryHDU | None = None):
        """Get the metadata of the fiducial image.

        Parameters
        ----------

        hdu : astropy.io.fits.PrimaryHDU | None, optional
            The Primary Header Data Unit of the FITS file. Can be provided
            to avoid consecutive opening and closing of the file.
            If set to ``None``, the file will be opened.
            Default is ``None``.

        Returns
        -------

        dict
            The metadata of the fiducial image as a dictionary.
        """

        hdu = hdu if hdu is not None else self.get_hdu()
        header = self.get_header(hdu=hdu)

        img_size = self.get_image(hdu=hdu).shape[0]
        cell_size = np.abs(header["CDELT1"] * 3600)

        return {
            "source_name": header["OBJECT"],
            "observatory": header["TELESCOP"],
            "img_size": img_size,
            "cell_size": cell_size,
            "fov": cell_size * img_size,
            "frequency": header["CRVAL3"],
            "bandwidth": header["CDELT3"],
            "wavelength": c.value / header["CRVAL3"],
            "src_ra": header["OBSRA"],
            "src_dec": header["OBSDEC"],
            "img_ra": header["CRVAL1"],
            "img_dec": header["CRVAL2"],
            "obs_date": header["DATE-OBS"],
            "flux_unit": header["BUNIT"],
            "beam": {
                "bmin": header["BMIN"] * 3600,
                "bmaj": header["BMAJ"] * 3600,
                "bpa": header["BPA"],
            },
        }

    def get_image(
        self,
        ra_incr_right: bool = True,
        flux_unit: str | None = None,
        allow_unit_prefixes: bool = True,
        hdu: PrimaryHDU | None = None,
    ):
        """Get the image data of the fiducial image.

        Parameters
        ----------

        ra_incr_right : bool, optional
            Determines whether the image should be returned
            with the Right Ascension (RA) increasing to the
            right like a normal coordinate axis.
            Default is ``True``.

        flux_unit : str, optional
            The flux density unit the image is supposed to have.
            Valid values are: ``'Jy/pix'``, ``"Jy/beam"`` or ``None``.
            The nominator of the unit (``'Jy'``) may also include prefixes like
            ``'m'`` for milli etc. if the parameter ``allow_unit_prefixes`` is set
            to ``True``. Otherwise an error will be raised.
            If set to ``None``, the unit from the FITS file will be used.
            Default is ``None``.

        allow_unit_prefixes : bool, optional
            Whether the Jansky unit (nominator of the ``flux_unit``) may include
            unit prefixes.
            Default is ``True``.

        hdu : astropy.io.fits.PrimaryHDU | None, optional
            The Primary Header Data Unit of the FITS file. Can be provided
            to avoid consecutive opening and closing of the file.
            If set to ``None``, the file will be opened.
            Default is ``None``.

        Returns
        -------

        numpy.ndarray
            The image data.
        """

        hdu = hdu if hdu is not None else self.get_hdu()

        image = hdu.data[0, 0].copy()
        header = self.get_header(hdu=hdu)

        if (header["CDELT1"] < 0 and ra_incr_right) or (
            header["CDELT1"] > 0 and not ra_incr_right
        ):
            image = image[:, ::-1]

        if flux_unit is None or flux_unit.lower() == header["BUNIT"].lower():
            return image

        flux_unit_components = flux_unit.split("/")
        flux_unit_nominator = units.Unit(flux_unit_components[0])

        if not allow_unit_prefixes and flux_unit_nominator != "Jy":
            raise ValueError(
                "The flux density unit may not contain prefixes if "
                "'allow_unit_prefixes=False'."
            )

        prev_flux_unit_nominator = units.Unit("Jy")

        image *= (1 * prev_flux_unit_nominator).to(flux_unit_nominator).value

        if flux_unit_components[1].lower() == header["BUNIT"].split("/")[1].lower():
            return image

        convert_args = dict(
            cell_size=np.abs(header["CDELT1"]), bmin=header["BMIN"], bmaj=header["BMAJ"]
        )

        match flux_unit_components[1]:
            case "pix":
                return beam2pix(image=image, **convert_args)
            case "beam":
                return pix2beam(image=image, **convert_args)
            case _:
                raise ValueError("The provided flux density unit is invalid!")

    def get_maximum_info(
        self,
        unit: str | units.Unit = "deg",
        hdu: PrimaryHDU | None = None,
    ):
        hdu = self.get_hdu() if hdu is not None else self.get_hdu()
        image = self.get_image(hdu=hdu)
        header = self.get_header(hdu=hdu)
        metadata = self.get_metadata(hdu=hdu)

        unit = units.Unit(unit)

        if unit.physical_type != "angle":
            raise ValueError("The given unit is not an angle unit (e.g. deg, arcsec)!")

        brightest_pixels = np.array(np.unravel_index(image.argmax(), image.shape))[::-1]
        brightest_unit = (
            (
                (brightest_pixels - np.array(image.shape)[::-1] // 2)
                * metadata["cell_size"]
                / 3600
                + np.array([header["CRVAL1"], header["CRVAL2"]])
            )
            * units.deg
        ).to(unit)

        return image.max(), brightest_pixels * units.pixel, brightest_unit

    def preprocess(
        self,
        clean: bool,
        remove_negatives: bool = True,
        ra_incr_right: bool = True,
        crop: tuple[list[float | None]] = ([None, None], [None, None]),
        rms_cut_args: dict | None = None,
        dbscan_args: dict | None = None,
        output_path: str | None = None,
        flux_unit: str | None = None,
        source_name: str | None = None,
        overwrite: bool = False,
    ):
        """Preprocesses the fiducial image by cropping and cleaning it.

        Parameters
        ----------

        clean : bool
            Whether to perform noise cleaning on the image.

        remove_negatives : bool, optional
            Whether to set negative values to zero.
            If ``clean=True`` this value is automatically ``True``.
            Default is ``True``.

        crop : tuple[list[float | None]], optional
            The crop of the image. This has to have the format
            ``([x_left, x_right], [y_left, y_right])``, where the left and right
            values for each axis are the upper and lower limits of the axes which
            should be shown.
            The limits have to create a quadratic image.

        rms_cut_args : dict, optional
            The arguments to pass to the
            ``radio_stats.cuts.dbscan_clean.rms_cut`` method.
            Default is ``{'sigma': 2.9}``.

        dbscan_args : dict, optional
            The arguments to pass to the
            ``radio_stats.cuts.dyn_range.dbscan_clean`` method.
            Default is ``{'min_brightness': 1e-4}``.

        output_path : str | None, optional
            The path to save the cleaned model at. If ``None`` same directory
            is used and name is appended a '_cleaned' suffix)
            Default is ``None``.

        flux_unit : str, optional
            The flux density unit in which the image should be saved.
            Available values are ``'Jy/beam'``, ``'Jy/pix'`` or ``None``.
            The unit may not contain unit prefixes!
            If set to ``None``, the unit from the original FITS file will be used.
            Default is ``None``.

        overwrite : bool, optional
            Whether to overwrite an existing file.
            Default is ``False``

        Returns
        -------

        Fiducial
            The processed ``Fiducial`` object.

        """

        if rms_cut_args is None:
            rms_cut_args = {"sigma": 2.9}

        if dbscan_args is None:
            dbscan_args = {"min_brightness": 1e-4}

        if output_path is None:
            processed_path = Path(
                f"{self._fits_path.parent / self._fits_path.stem}_processed.fits"
            ).resolve()
        else:
            processed_path = Path(output_path).resolve()

        if processed_path.is_file() and not overwrite:
            warnings.warn(
                "The file already exists and is not supposed to be overwritten."
                "Skipping processing.",
                stacklevel=1,
            )
            return Fiducial(fits_path=processed_path)

        hdu = self.get_hdu()

        fiducial = self.get_image(
            ra_incr_right=ra_incr_right,
            flux_unit=flux_unit,
            hdu=hdu,
            allow_unit_prefixes=False,
        )

        header = self.get_header(hdu=hdu)
        metadata = self.get_metadata(hdu=hdu)

        header["CDELT1"] = (
            np.abs(header["CDELT1"]) if ra_incr_right else -np.abs(header["CDELT1"])
        )
        header["BUNIT"] = flux_unit

        if source_name is not None:
            header["OBJECT"] = source_name

        if clean:
            fiducial = rms_cut(fiducial, **rms_cut_args)
            fiducial = dbscan_clean(fiducial, **dbscan_args)
        elif remove_negatives:
            fiducial[fiducial < 0] = 0

        img_size = metadata["img_size"]
        cell_size = metadata["cell_size"]

        crop_max = np.array([[0, img_size], [0, img_size]])
        crop = np.array(crop)

        for i in range(2):
            for j in range(2):
                crop[i, j] = crop[i, j] if crop[i, j] is not None else crop_max[i, j]

        if crop[0][1] - crop[0][0] != crop[1][1] - crop[1][0]:
            raise IndexError("The given crop limits have to create a quadratic image!")

        fiducial = fiducial[crop[1][0] : crop[1][1], crop[0][0] : crop[0][1]]
        center_pixels = ((crop[0][0] + crop[0][1]) // 2, (crop[1][0] + crop[1][1]) // 2)

        # Adjust physical center of image
        header["CRVAL1"] = (
            metadata["img_ra"] + (center_pixels[0] - img_size // 2) * cell_size / 3600
        ) % 360

        header["CRVAL2"] = (
            metadata["img_dec"]
            + (center_pixels[1] - img_size // 2) * cell_size / 3600
            + 90
        ) % 180 - 90

        hdu.data = fiducial[None, None]
        hdu.writeto(processed_path, overwrite=overwrite)

        return Fiducial(fits_path=processed_path)

    def plot(
        self,
        display_beam: bool = True,
        ra_incr_right: bool = True,
        ax_unit: str | units.Unit = "pixel",
        use_relative_ax: bool = True,
        flux_unit: str | None = None,
        display_title: bool = True,
        norm: str | matplotlib.colors.Normalize | None = None,
        colorbar_shrink: float = 1,
        cmap: str | matplotlib.colors.Colormap | None = "inferno",
        plot_args: dict = None,
        fig_args: dict = None,
        save_to: str | None = None,
        save_args: dict = None,
        fig: matplotlib.figure.Figure | None = None,
        ax: matplotlib.axes.Axes | None = None,
    ):
        """Plots the uncleaned model

        Parameters
        ----------
        display_beam : bool, optional
            Whether to display size and orientation of the beam as
            a white ellipse in the lower left of the image.
            Default is ``True``.
        ra_incr_right : bool, optional
            Whether the Right Ascension coordinate (x-axis) should
            increase to the right.
            Default is ``True``.
        ax_unit : str | astropy.units.Unit, optional
            The unit in which to show the ticks of the x and y-axes in.
            The y-axis is the Declination (DEC) and the x-axis is the
            Right Ascension (RA).
            The unit has to be given as a string or an ``astropy.units.Unit``.
            The string must correspond to the string representation of an
            ``astropy.units.Unit``.

            Valid units are either ``pixel`` or angle units like ``arcsec``, ``degree``
            etc.. Default is ``pixel``.
        use_relative_ax : bool, optional
            Whether to display the coordinate relative to the center of the image
            (not the source position).
            Default is ``True``.
        flux_unit : str | None, optional
            The flux density unit the image is supposed to have.
            Valid values are: ``'Jy/pix'``, ``"Jy/beam"`` or ``None``.
            The nominator of the unit (``'Jy'``) may also include prefixes like
            ``'m'`` for milli etc..
            If set to ``None``, the unit from the FITS file will be used.
            Default is ``None``.
        display_title: bool, optional
            Whether to display the name of the source as the title of the plot.
            Default is ``True``.
        norm: str | matplotlib.colors.Normalize | None, optional
            The name of the norm or the norm itself.
            Possible values are:

            - ``log``:          Returns a logarithmic norm with clipping on (!), meaning
                                values above the maximum will be mapped to the maximum
                                and values below the minimum will be mapped to the
                                minimum, thus avoiding the appearance of a colormaps
                                'over' and 'under' colors (e.g. in case of negative
                                values). Depending on the use case this is desirable but
                                in case that it is not, one can set the norm to
                                ``log_noclip`` or provide a custom norm.

            - ``log_noclip``:   Returns a logarithmic norm with clipping off.

            - ``centered``:     Returns a linear norm which centered around zero.

            - ``sqrt``:         Returns a power norm with exponent 0.5, meaning the
                                square-root of the values.

            - other:            A value not declared above will be returned as is,
                                meaning that this could be any value which exists in
                                matplotlib itself.

            Default is ``None``, meaning no norm will be applied.
        colorbar_shrink: float, optional
            The shrink parameter of the colorbar. This can be needed if the plot is
            included as a subplot to adjust the size of the colorbar.
            Default is ``1``, meaning original scale.
        cmap: str | matplotlib.colors.Colormap, optional
            The colormap to be used for the plot.
            Default is ``'inferno'``.
        plot_args : dict, optional
            The additional arguments passed to the scatter plot.
            Default is ``{"color":"royalblue"}``.
        fig_args : dict, optional
            The additional arguments passed to the figure.
            If a figure object is given in the ``fig`` parameter, this
            value will be discarded.
            Default is ``{}``.
        save_to : str | None, optional
            The name of the file to save the plot to.
            Default is ``None``, meaning the plot won't be saved.
        save_args : dict, optional
            The additional arguments passed to the ``fig.savefig`` call.
            Default is ``{"bbox_inches":"tight"}``.
        fig : matplotlib.figure.Figure | None, optional
            A custom figure object.
            If set to ``None``, the ``ax`` parameter also has to be ``None``!
            Default is ``None``.
        ax : matplotlib.axes.Axes | None, optional
            A custom axes object.
            If set to ``None``, the ``fig`` parameter also has to be ``None``!
            Default is ``None``.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        ax : matplotlib.axes.Axes
            The axes object.
        """

        if plot_args is None:
            plot_args = {}

        if fig_args is None:
            fig_args = {}

        if save_args is None:
            save_args = dict(bbox_inches="tight")

        fig, ax = _configure_axes(fig=fig, ax=ax, fig_args=fig_args)
        norm = _get_norm(norm) if isinstance(norm, str) else norm

        unit = units.Unit(ax_unit)

        hdu = self.get_hdu()
        metadata = self.get_metadata(hdu=hdu)

        flux_unit = flux_unit if flux_unit is not None else metadata["flux_unit"]

        image = self.get_image(flux_unit=flux_unit, ra_incr_right=True, hdu=hdu)

        img_size = image.shape[0]
        cell_size = metadata["cell_size"]

        if unit.physical_type == "angle":
            extent = (
                np.array([-img_size / 2, img_size / 2] * 2)
                * cell_size
                * units.arcsecond
            ).to(unit)

            if use_relative_ax:
                label_prefix = "Relative "

            else:
                center_pos = (
                    np.array([metadata["img_ra"], metadata["img_dec"]]) * units.degree
                ).to(unit)
                extent[:2] += center_pos[0]
                extent[2:] += center_pos[1]
                label_prefix = ""

            ax.set_xlabel(f"{label_prefix}Right Ascension / {unit}")
            ax.set_ylabel(f"{label_prefix}Declination / {unit}")

            extent = extent.value

        else:
            if unit != units.pixel:
                warnings.warn(
                    f"The given unit {unit} is no angle unit! Using pixels instead.",
                    stacklevel=2,
                )

            unit = units.pixel

            extent = None

            ax.set_xlabel("Pixels")
            ax.set_ylabel("Pixels")

        im = ax.imshow(
            image,
            norm=norm,
            cmap=cmap,
            interpolation="none",
            origin="lower",
            extent=extent,
            **plot_args,
        )
        ax.xaxis.set_inverted(not ra_incr_right)

        ax_formatter = ScalarFormatter(useOffset=False)
        ax.xaxis.set_major_formatter(ax_formatter)
        ax.yaxis.set_major_formatter(ax_formatter)

        if display_beam:
            beam_info = metadata["beam"]

            bmin = beam_info["bmin"] * units.arcsecond
            bmaj = beam_info["bmaj"] * units.arcsecond
            bpa = beam_info["bpa"]

            position = np.array([img_size * 0.1] * 2)

            if not ra_incr_right:
                position[0] = img_size * 0.9

            if unit != units.pixel:
                bmin = bmin.to(unit).value
                bmaj = bmaj.to(unit).value
                position *= cell_size
                position = (position * units.arcsecond).to(unit).value
                position += np.array([extent[0], extent[2]])
            else:
                bmin = bmin.value / cell_size
                bmaj = bmaj.value / cell_size

            ax.add_patch(
                Ellipse(
                    xy=position,
                    width=bmin,
                    height=bmaj,
                    angle=bpa,
                    facecolor="white",
                ),
            )

        fig.colorbar(
            im,
            ax=ax,
            shrink=colorbar_shrink,
            label=f"Flux Density / {flux_unit}",
        )

        if display_title:
            ax.set_title(metadata["source_name"])

        if save_to is not None:
            fig.savefig(save_to, **save_args)

        return fig, ax


def _configure_axes(
    fig: matplotlib.figure.Figure | None,
    ax: matplotlib.axes.Axes | None,
    fig_args: dict = None,
):
    """Configures figure and axis depending if they were given
    as parameters.

    If neither figure nor axis are given, a new subplot will be created.
    If they are given the given ones will be returned.
    If only one of both is not given, this will cause an exception.

    Parameters
    ----------
    fig : matplotlib.figure.Figure | None
        The figure object.
    ax : matplotlib.axes.Axes | None
        The axes object.
    fig_args : dict, optional
        Optional arguments to be supplied to the ``plt.subplots`` call.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.
    """
    if fig_args is None:
        fig_args = {}

    if None in (fig, ax) and not all(x is None for x in (fig, ax)):
        raise KeyError("The parameters ax and fig have to be both None or not None!")

    if ax is None:
        fig, ax = plt.subplots(layout="tight", **fig_args)

    return fig, ax


def _get_norm(norm: str):
    """Converts a string parameter to a matplotlib norm.

    Parameters
    ----------
    norm : str
        The name of the norm.
        Possible values are:

        - ``log``:          Returns a logarithmic norm with clipping on (!), meaning
                            values above the maximum will be mapped to the maximum and
                            values below the minimum will be mapped to the minimum, thus
                            avoiding the appearance of a colormaps 'over' and 'under'
                            colors (e.g. in case of negative values). Depending on the
                            use case this is desirable but in case that it is not, one
                            can set the norm to ``log_noclip`` or provide a custom norm.

        - ``log_noclip``:   Returns a logarithmic norm with clipping off.

        - ``centered``:     Returns a linear norm which centered around zero.

        - ``sqrt``:         Returns a power norm with exponent 0.5, meaning the
                            square-root of the values.

        - other:            A value not declared above will be returned as is, meaning
                            that this could be any value which exists in matplotlib
                            itself.

    Returns
    -------
    matplotlib.colors.Normalize | str
        The norm or the str if no specific norm is defined for the string.
    """
    match norm:
        case "log":
            return matplotlib.colors.LogNorm(clip=True)
        case "log_noclip":
            return matplotlib.colors.LogNorm(clip=False)
        case "centered":
            return matplotlib.colors.CenteredNorm()
        case "sqrt":
            return matplotlib.colors.PowerNorm(0.5)
        case _:
            return norm
