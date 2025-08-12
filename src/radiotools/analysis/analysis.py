import torch
from astropy.io import fits
from numpy.typing import ArrayLike

from radiotools.utils import rms


def get_source_rms(image: ArrayLike, center: tuple[int, int], *, offset=75):
    """Get the root mean square (RMS) of a given source.

    Parameters
    ----------
    image : array_like, shape (N, M)
        Array containing image data of the source.
    center : tuple[int, int]
        Tuple of the center coordinates where the RMS
        is calculated for a region spanned by ``offset``.
    offset : int, optional
        Offset that spans a region around ``center``:

        .. code::
            x0 = center_x - offset
            x1 = center_x + offset

            y0 = center_y - offset
            y1 = center_y + offset

        Default: 75

    Returns
    -------
    rms : float
        RMS for the given source.
    """
    center_x = int(center[0])
    center_y = int(center[1])

    x0 = center_x - offset
    x1 = center_x + offset

    y0 = center_y - offset
    y1 = center_y + offset

    _rms = torch.from_numpy(rms(image[x0:x1, y0:y1]))

    return torch.mean(_rms)


def dynamic_range(path: str, *, offset: int = 75):
    """Computes the dynamic range of a source.

    Parameters
    ----------
    path : str or Path
        Path to a FITS file containing image data of the source.
    offset : int, optional
        Offset for the region around the center pixel of the source.
        Used to compute the RMS value of the source. Default: 75

    Returns
    -------
    dr : float
        Dynamic range of the source.
    """
    hdu = fits.open(path)

    image = hdu[0].data[0, 0, ...]
    center = hdu[0].header["CRPIX1"], hdu[0].header["CRPIX2"]

    _rms = get_source_rms(image, center, offset=75)

    dr = image.max() / _rms

    return dr
