import re

import numpy as np
import requests
from bs4 import BeautifulSoup
from numpy.typing import ArrayLike


def get_array_names(url: str) -> list[str]:
    """Fetches array names from a given URL
    leading to a directory.

    Parameters
    ----------
    url : str
        URL leading to a directory containing layout
        txt files.

    Returns
    -------
    layouts : list[str]
        List of available layouts.
    """
    r = requests.get(url)
    soup = BeautifulSoup(r.text, features="html.parser")

    layouts = list(set(re.findall(r"\b\w+\.txt\b", str(soup.div))))

    return layouts


def rms(a: ArrayLike, *, axis: int | None = 0):
    """Return an array of the root-mean-square (RMS) value of
    the passed array.

    Parameters
    ----------
    a : array_like
        Array of which to find the RMS.
    axis : int or None

    Returns
    -------
    rms : np.ndarray
        Array of rms values.
    """
    if np.ndim(a) == 0:
        axis = None

    return np.sqrt(np.mean(a**2, axis=axis))


def beam2pix(
    image: ArrayLike,
    cell_size: float,
    bmin: float,
    bmaj: float,
):
    """Converts an image from Jy/beam to Jy/px.

    Parameters
    ----------
    image : array_like
        Input image that is to be converted.
    cell_size : float
        The physical size of one pixel in arcseconds.
    bmin : float
        The minor axis of the beam in arcseconds.
    bmaj : float
        The major axis of the beam in arcseconds.

    Returns
    -------
    array_like
        Converted image in units of Jy/pix.
    """

    return image * (4 * np.log(2) * np.power(cell_size, 2) / (np.pi * bmin * bmaj))


def pix2beam(
    image: ArrayLike,
    cell_size: float,
    bmin: float,
    bmaj: float,
):
    """Converts an image from Jy/pix to Jy/beam.

    Parameters
    ----------
    image : array_like
        Input image that is to be converted.
    cell_size : float
        The physical size of one pixel in arcseconds.
    bmin : float
        The minor axis of the beam in arcseconds.
    bmaj : float
        The major axis of the beam in arcseconds.

    Returns
    -------
    array_like
        Converted image in units of Jy/beam.
    """

    return image / (4 * np.log(2) * np.power(cell_size, 2) / (np.pi * bmin * bmaj))
