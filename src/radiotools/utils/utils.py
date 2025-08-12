import numpy as np
import requests
from astropy.io import fits
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

    a_tags = soup.find_all("a")

    layouts = []
    for i in a_tags:
        if ".txt" in str(i.get("href")):
            layouts.append(i.get("aria-label").split(".txt")[0])

    layouts = list(set(layouts))

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


def img2jansky(image: ArrayLike, header: fits.Header):
    """Converts an image from Jy/beam to Jy/px.

    Parameters
    ----------
    image : array_like
        Input image that is to be converted.
    header : :class:`astropy.io.fits.header.Header`
        FITS file header belonging to the respective image.

    Returns
    -------
    array_like
        Converted image in units of Jy/px.
    """
    return (
        4
        * image
        * np.log(2)
        * np.power(header["CDELT1"], 2)
        / (np.pi * header["BMIN"] * header["BMAJ"])
    )
