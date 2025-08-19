import re

import numpy as np
import requests
import subprocess
from astropy.io import fits
from bs4 import BeautifulSoup
from numpy.typing import ArrayLike
from pathlib import Path


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


def uvfits2ms(fits_path, ms_path):
    """Converts a uvfits_file into a measurement set (ms file).

    Parameters
    ----------
        fits_path : str
            path to fits_file
        ms_path : str
            path to store ms_file
    """
    casa = "casa --quiet --nologger --nologfile --nogui --norc --agg -c "
    arg = (
        "importuvfits(fitsfile='"
        + str(fits_path)
        + "'"
        + ", vis="
        + "'"
        + str(ms_path)
        + "'"
        + ")"
    )

    command = casa + '"' + arg + '"'

    subprocess.run(command, shell=True)


def rmtree(root: Path):
    """Recursively remove directories and files
    starting from root directory.

    Parameters
    ----------
    root : Path
        Root path of the directories you want to delete.
    """
    for p in root.iterdir():
        if p.is_dir():
            rmtree(p)
        else:
            p.unlink()

    root.rmdir()
