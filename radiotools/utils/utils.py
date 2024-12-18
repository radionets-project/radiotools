import json

import numpy as np
import requests
from astropy.io import fits
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
    text = json.loads(r.text)

    layouts = [
        t["path"].split(".txt")[0].split("/")[-1]
        for t in text["tree"]
        if "layouts" and ".txt" in t["path"]
    ]

    return layouts


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
