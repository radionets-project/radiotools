import shutil
import uuid
import warnings
from pathlib import Path

import pandas as pd
from astropy.io import fits
from casatools import ms as MeasurementTool


class Measurement:
    """
    A tool to convert between FITS files and NRAO CASA measurement sets
    """

    class ObservationTime:

        """
        A subclass to format and offset the start time of an observation

        Parameters
        ----------
        time: pandas.datetime
        The start time as a pandas datetime object
        """

        def __init__(self, time):
            self._obs_time = time

        """
            Formats the Observation time as a string

            Parameters
            ----------
            fmt: str, optional
            The format of the ObservationTime (e.g. `%d-%m-%Y %H:%M:%S`)
        """

        def format(self, fmt="%d-%m-%Y %H:%M:%S"):
            return self._obs_time.strftime(fmt)

        """
            Returns a copy of the current time with a given offset

            Parameters
            ----------
            hours: int, optional
            The hour offset

            minutes: int, optional
            The minute offset

            seconds: int, optional
            The second offset
        """

        def offset(self, hours=0, minutes=0, seconds=0):
            return Measurement.ObservationTime(
                self._obs_time
                + pd.Timedelta(hours=hours, minutes=minutes, seconds=seconds)
            )

    def __init__(self):
        None

    """
        Returns the time at which the Observation was started.
    """

    def get_obs_time(self):
        return Measurement.ObservationTime(
            pd.to_datetime(self.get_fits()[0].header["DATE-OBS"])
        )

    """
           Saves the current measurement
           If the Measurement was not not created using a CASA measurement set,
           this will raise an exception.

           Parameters
           ----------
           path: str
           The path of the root of the measurement set

           overwrite: bool, optional
           Whether an existing measurement set should be overwritten.
    """

    def save_as_ms(self, path, overwrite=False):
        root = Path(path)

        if root.exists():
            if overwrite:
                shutil.rmtree(root)
            else:
                warnings.warn(
                    f"The directory {root} already exists! If you want to overwrite it set overwrite=True!"
                )

        if not root.exists() or overwrite:
            ms = MeasurementTool()
            ms.fromfits(msfile=path, fitsfile=self._fits_path, nomodify=not overwrite)

    """
               Saves the current measurement as a FITS file
               If the Measurement was not not created using a FITS file,
               this will raise an exception.

               Parameters
               ----------
               path: str
               The path of the FITS file

               overwrite: bool, optional
               Whether an existing FITS file should be overwritten.
    """

    def save_as_fits(self, path, overwrite=False):
        file = Path(path)

        if file.is_file():
            if overwrite:
                file.unlink()
            else:
                warnings.warn(
                    f"The file {file} already exists! If you want to overwrite it set overwrite=True!"
                )

        if not file.is_file() or overwrite:
            self._ms.tofits(path, overwrite=overwrite)

    """
            Returns a copy of the current measurement as a NRAO CASA measurement set.
            There will be no permanent physical version of the measurement on the disk.
            If the Measurement was not not created using a FITS file,
            this will try to return a measurement set saved in this Measurement.
            If no measurement set is present, this will raise an exception.
    """

    def get_ms(self):
        if not hasattr(self, "_fits_path"):
            return self._ms

        def gen_rng_dir():
            return Path(f"./temp_fits_{uuid.uuid4()}/")

        temp_path = gen_rng_dir()

        while temp_path.exists():
            temp_path = gen_rng_dir()

        self.save_as_ms(temp_path.absolute(), overwrite=False)

        ms = MeasurementTool()
        ms.open(temp_path)

        shutil.rmtree(temp_path)

        return ms

    """
        Returns a copy of the current measurement as a FITS file.
        If the Measurement was not not created using a CASA measurement set,
        this will try to return a FITS file saved in this Measurement.
        If no FITS file is present, this will raise an exception.
    """

    def get_fits(self):
        if not hasattr(self, "_ms_path"):
            return self._fits

        def gen_rng_file():
            return Path(f"./temp_fits_{uuid.uuid4()}.fits")

        temp_path = gen_rng_file()

        while temp_path.is_file():
            temp_path = gen_rng_file()

        self.save_as_fits(str(temp_path), overwrite=False)

        _fits = fits.open(temp_path)
        temp_path.unlink()

        return _fits

    """
        Creates a Measurement from a FITS file

        Parameters
        ----------
        fits_path: str
        The path to the FITS file
    """

    @classmethod
    def from_fits(cls, fits_path):
        cls = cls()
        cls._fits_path = fits_path

        cls._fits = fits.open(fits_path)

        return cls

    """
    Creates a Measurement from a NRAO CASA measurement set

    Parameters
    ----------
    ms_path: str
    The path to the root of the measurement set

    """

    @classmethod
    def from_ms(cls, ms_path):
        cls = cls()

        ms = MeasurementTool()
        ms.open(ms_path)
        cls._ms_path = ms_path
        cls._ms = ms

        return cls