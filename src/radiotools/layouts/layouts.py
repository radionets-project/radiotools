import urllib
import uuid
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import EarthLocation
from casacore.tables import table
from numpy.typing import ArrayLike

pd.options.display.float_format = "{:f}".format


class Layout:
    """
    A tool to convert radio telescope array layout config files between different types.

    """

    def get_baselines(self):
        """Returns an array containing the lengths of all
        unique (!) baselines in meters.
        """
        loc = np.array([self.x, self.y]).T
        baselines = np.array([])

        for i, vec1 in enumerate(loc):
            for vec2 in loc[i + 1 :]:
                baselines = np.append(
                    baselines,
                    np.linalg.norm(np.reshape(vec1 - vec2, (2, 1)), ord=2, axis=0),
                )

        return baselines

    def get_baseline_vecs(self):
        """Returns an array containing the vectors of all (!)
        baselines (including conjugates).
        """
        loc = np.array([self.x, self.y]).T
        baselines = np.array([])

        for vec1 in loc:
            for vec2 in loc:
                baselines = np.append(baselines, np.reshape(vec1 - vec2, (2, 1)))

        return np.reshape(baselines, (int(len(baselines) / 2), 2)).T

    def get_max_resolution(self, frequency):
        """
        Returns the maximal resolution of the layout
        at a given frequency in arcsec / px.

        Parameters
        ----------
        frequency : float or array_like
            The frequency at which the array is observing

        """
        return 3600 * 180 / np.pi * 3 * 1e8 / (frequency * np.max(self.get_baselines()))

    def get_station(self, name):
        """Returns all information about the station
        with the given name as a `pandas.Series`.

        Parameters
        ----------
        name : str
            The name of the station (antenna).
            This is case sensitive!

        Returns
        -------
        pd.Series
            Pandas series containing all information
            about the given station.
        """
        if name not in self.names:
            raise KeyError(
                "This station could not be found. Make sure "
                "you typed the name correctly (case sensitive)!"
            )

        df = self.get_dataframe()

        return df.loc[df["station_name"] == name]

    def get_dataframe(self):
        """Returns the layout data as a :class:`pandas.DataFrame`."""

        return pd.DataFrame(
            data={
                "x": self.x,
                "y": self.y,
                "z": self.z,
                "dish_dia": self.dish_dia,
                "station_name": self.names,
                "el_low": self.el_low,
                "el_high": self.el_high,
                "sefd": self.sefd,
                "altitude": self.altitude,
            }
        )

    def __str__(self):
        output = f"Configuration file loaded from: {self.cfg_path}"
        output += f"\nRelative to site: {self.rel_to_site}"
        output += f"\nNumber of antennas: {len(self.x)}"
        output += f"\nNumber of baselines: {np.sum([i for i in range(len(self.x))])}"
        output += f"\nLongest baseline: {np.max(self.get_baselines())} m"
        output += f"\nShortest baseline: {np.min(self.get_baselines())} m"
        output += f"\n\n{self.get_dataframe()}"

        return output

    def display(self):
        """
        Prints all information contained in the layout

        """
        print(self.__str__())

    def is_relative(self):
        return not (self.rel_to_site is None or self.rel_to_site == "")

    def as_relative(self, rel_to_site):
        """Returns a copy of the current layout in
        relative coordinates.

        Parameters
        ----------
        rel_to_site : str
            The name of the site the coordinates are
            supposed to be relative to. Has to be an
            existing site for
            :func:`astropy.coordinates.EarthLocation.of_site()`.
        """

        def gen_rng_file():
            return Path(f"./temp_cfg_{uuid.uuid4()}.cfg")

        temp_path = gen_rng_file()

        while temp_path.is_file():
            temp_path = gen_rng_file()

        self.save(temp_path, rel_to_site=rel_to_site)
        new_layout = Layout.from_pyvisgen(temp_path, rel_to_site=rel_to_site)

        temp_path.unlink()
        new_layout.cfg_path = self.cfg_path

        return new_layout

    def as_absolute(self):
        """Returns a copy of the current layout in absolute
        coordinates. Requires the layout to be in relative
        coordinates.
        """

        if not self.is_relative():
            raise TypeError(
                "This layout is not saved in relative coordinates and can therefore "
                "not be converted in absolute coordinates."
            )

        def gen_rng_file(increment=0):
            return Path(f"./temp_cfg_{uuid.uuid4()}.cfg")

        temp_path = gen_rng_file()

        i = 1
        while temp_path.is_file():
            temp_path = gen_rng_file(i)
            i += 1

        self.save(temp_path)
        new_layout = Layout.from_pyvisgen(temp_path)

        temp_path.unlink()
        new_layout.cfg_path = self.cfg_path

        return new_layout

    def plot_uv(
        self,
        save_to_file="",
        ref_frequency=None,
        plot_args=None,
        save_args=None,
        show_zeros=False,
    ):
        """Plots the uv-sampling (uv-plane) of the array.

        Parameters
        ----------
        save_to_file : str, optional
            The name of the file the plot should be saved to.
        plot_args : dict, optional
            Arguments to pass to the axis.scatter function
        save_args : dict, optional
            Arguments to pass to the figure.savefig function
        """
        if plot_args is None:
            plot_args = {"color": "royalblue", "alpha": 0.5}

        if save_args is None:
            save_args = {}

        baselines = self.get_baseline_vecs()

        if not show_zeros:
            nonzero_bl = np.linalg.norm(baselines, axis=0) != 0
            baselines = baselines[:, nonzero_bl]

        fig, ax = plt.subplots(1, 1, layout="constrained")

        if ref_frequency is not None:
            baselines /= 3e8 / ref_frequency

        ax.scatter(baselines[0], baselines[1], **plot_args)
        ax.set_xlabel("$u$ in m" if ref_frequency is None else "$u/\\lambda$")
        ax.set_ylabel("$v$ in m" if ref_frequency is None else "$v/\\lambda$")

        if save_to_file != "":
            fig.savefig(save_to_file, **save_args)

        return fig, ax

    def plot(
        self,
        save_to_file="",
        annotate=False,
        limits=None,
        plot_args=None,
        save_args=None,
    ):
        """Generates a plot of the arrangement of the layout.

        Parameters
        ----------
        save_to_file : str, optional
            The name of the file the plot should be saved to.
        annotate : bool, optional
            Whether to mark the stations with their respective names.
        limits : tuple of tuples, optional
            The x and y bounds (e.g. `((0,1), (0,1))`). Set tuple of one
            axis (x or y) to None to only limit the other axis.
        plot_args : dict, optional
            Arguments to pass to the axis.scatter function
        save_args : dict, optional
            Arguments to pass to the figure.savefig function
        """
        if plot_args is None:
            plot_args = {}

        if save_args is None:
            save_args = {}

        singular_alt = len(np.unique(self.altitude)) == 1

        options = {
            "color": "#f54254" if singular_alt else None,
            "cmap": "cividis" if not singular_alt else None,
            "c": self.altitude if not singular_alt else None,
        }

        fig, ax = plt.subplots(1, 1)

        im = ax.scatter(self.x, self.y, **options, **plot_args)

        if limits:
            if limits[0]:
                ax.set(xlim=limits[0])
            if limits[1]:
                ax.set(ylim=limits[1])

        if not singular_alt:
            fig.colorbar(im, ax=ax, label="Altitude")

        if annotate:
            for _, row in self.get_dataframe().iterrows():
                ax.annotate(
                    text=f"{row['station_name']}", xy=(row.x, row.y), fontsize=8
                )

        ax.set_xlabel(f"{'Relative' if self.is_relative() else 'Geocentric'} $x$ in m")
        ax.set_ylabel(f"{'Relative' if self.is_relative() else 'Geocentric'} $y$ in m")
        ax.set_box_aspect(1)

        if save_to_file != "":
            fig.savefig(save_to_file, **save_args)

        return fig, ax

    def save(self, path, fmt="pyvisgen", overwrite=False, rel_to_site=None):
        """
        Saves the layout to a layout file.

        Parameters
        ----------
        path : str
            The path of the file to save the array layout to.
        fmt : str, optional
            The layout format the output file is supposed to have
            (available: casa, pyvisgen) (default is pyvisgen).
        overwrite : bool, optional
            Whether to overwrite the file if it already exists
            (default is False).
        rel_to_site : str, optional
            The name of the site the coordinates are supposed to be saved relative to.
            Is ignored if `None` or empty or `fmt` is not set to 'pyvisgen'.
            Has to be an existing site for
            :func:`astropy.coordinates.EarthLocation.of_site()`.
        """

        FORMATS = ["casa", "pyvisgen"]

        file = Path(path)

        if file.exists():
            if overwrite:
                file.unlink()
            else:
                raise FileExistsError(
                    f"The file {file} already exists! If you want "
                    "to overwrite it set overwrite=True!"
                )

        data = []

        save_relative = not (rel_to_site is None or rel_to_site == "")

        nx, ny, nz = self.x, self.y, self.z

        if save_relative:
            # Is supposed to saved be in relative (local tangent plane) coordinates

            location = EarthLocation.of_site(rel_to_site)

            if self.is_relative():
                # ... and is already relative --> reconvert
                # to absolute (if not same site)

                prev_location = EarthLocation.of_site(self.rel_to_site)
                nx, ny, nz = itrf2loc(
                    *loc2itrf(
                        prev_location.x.value,
                        prev_location.y.value,
                        prev_location.z.value,
                        self.x,
                        self.y,
                        self.z,
                    ),
                    location.x.value,
                    location.y.value,
                    location.z.value,
                )
            else:
                nx, ny, nz = itrf2loc(
                    self.x,
                    self.y,
                    self.z,
                    location.x.value,
                    location.y.value,
                    location.z.value,
                )

        else:
            # Is supposed to be saved in absolute (geocentric) coordinates

            if self.is_relative():
                # ... but is relative --> convert to absolute
                prev_location = EarthLocation.of_site(self.rel_to_site)
                nx, ny, nz = loc2itrf(
                    prev_location.x.value,
                    prev_location.y.value,
                    prev_location.z.value,
                    self.x,
                    self.y,
                    self.z,
                )

        match fmt:
            case "pyvisgen":
                data.append(
                    "station_name X Y Z dish_dia el_low el_high SEFD altitude\n"
                )

                for i in range(0, len(self.x)):
                    row = map(
                        str,
                        [
                            self.names[i],
                            nx[i],
                            ny[i],
                            nz[i],
                            self.dish_dia[i],
                            self.el_low[i],
                            self.el_high[i],
                            self.sefd[i],
                            self.altitude[i],
                        ],
                    )
                    data.append(" ".join(row) + "\n")

            case "casa":
                if save_relative:
                    data.append(f"observatory={rel_to_site}")
                    data.append("coordsys=LOC (local tangent plane)")

                data.append("# X Y Z dish_dia station_name\n")

                for i in range(0, len(self.x)):
                    row = map(
                        str,
                        [
                            nx[i],
                            ny[i],
                            nz[i],
                            self.dish_dia[i],
                            self.names[i],
                        ],
                    )
                    data.append(" ".join(row) + "\n")

            case _:
                raise ValueError(
                    f"{fmt} is not a valid format! Possible "
                    f"formats are: {', '.join(FORMATS)}!"
                )

        with open(file, "w", encoding="utf-8") as f:
            f.writelines(data)

    @classmethod
    def from_casa(
        cls, cfg_path, el_low=15, el_high=85, sefd=0, altitude=0, rel_to_site=None
    ):
        """
        Import a layout from a NRAO CASA layout config.

        Parameters
        ----------
        cfg_path : str
            The path of the config file to import.
        el_low : float or array_like, optional
            The minimal elevation in degrees the telescope can be adjusted to.
            If provided as singular number all telescopes in the array will
            be assigned the same value.
        el_high : float or array_like, optional
            The maximal elevation in degrees the telescope can be adjusted to.
            If provided as singular number all telescopes in the array will
            be assigned the same value.
        sefd : float or array_like, optional
            The system equivalent flux density of the telescope.
            If provided as singular number all telescopes in the array will
            be assigned the same value.
        altitude : float or array_like, optional
            The altitude of the telescope.
            If provided as singular number all telescopes in the array will
            be assigned the same value.
        rel_to_site : str, optional
            The name of the site the coordinates are relative to.
            Is ignored if `None` or empty.
            Has to be an existing site for
            :func:`astropy.coordinates.EarthLocation.of_site()`.
        """

        df = pd.read_csv(
            cfg_path,
            delimiter="\s+",
            encoding="utf-8",
            skip_blank_lines=True,
            names=["x", "y", "z", "dish_dia", "station_name"],
            dtype={
                "x": float,
                "y": float,
                "z": float,
                "dish_dia": float,
                "station_name": str,
            },
            comment="#",
        )
        cls = cls()
        cls.cfg_path = cfg_path
        cls.rel_to_site = rel_to_site
        cls.x = df["x"]
        cls.y = df["y"]
        cls.z = df["z"]
        cls.dish_dia = df["dish_dia"]
        cls.names = df["station_name"]
        cls.el_low = np.repeat(el_low, len(cls.x)) if np.isscalar(el_low) else el_low
        cls.el_high = (
            np.repeat(el_high, len(cls.x)) if np.isscalar(el_high) else el_high
        )
        cls.sefd = np.repeat(sefd, len(cls.x)) if np.isscalar(sefd) else sefd
        cls.altitude = (
            np.repeat(altitude, len(cls.x)) if np.isscalar(altitude) else altitude
        )

        return cls

    @classmethod
    def from_pyvisgen(cls, cfg_path, rel_to_site=None):
        """Import a layout from a radionets pyvisgen layout config.

        Parameters
        ----------
        cfg_path : str
            The path of the config file to import.

        rel_to_site : str, optional
            The name of the site the coordinates are relative to.
            Is ignored if `None` or empty.
            Has to be an existing site for
            :func:`astropy.coordinates.EarthLocation.of_site()`.
        """

        df = pd.read_csv(
            cfg_path,
            delimiter="\s+",
            encoding="utf-8",
            skip_blank_lines=True,
            dtype={
                "station_name": str,
                "x": float,
                "y": float,
                "z": float,
                "dish_dia": float,
                "el_low": float,
                "el_high": float,
                "sefd": float,
                "altitude": float,
            },
        )
        df.columns = map(str.lower, df.columns)

        cls = cls()
        cls.names = df["station_name"]
        cls.cfg_path = cfg_path
        cls.rel_to_site = rel_to_site
        cls.x = df["x"]
        cls.y = df["y"]
        cls.z = df["z"]
        cls.dish_dia = df["dish_dia"]
        cls.el_low = df["el_low"]
        cls.el_high = df["el_high"]
        cls.sefd = df["sefd"]
        cls.altitude = df["altitude"]

        return cls

    @classmethod
    def from_measurement_set(
        cls,
        root_path: str,
        sefd: int | ArrayLike,
        altitude: int | ArrayLike,
        rel_to_site: str | None = None,
    ):
        antennas = table(root_path + "/ANTENNA/", ack=False)

        positions = antennas.getcol("POSITION")
        stations = antennas.getcol("STATION")
        dish_diameters = antennas.getcol("DISH_DIAMETER")

        df = pd.DataFrame(
            data={
                "station_name": stations,
                "x": positions[0],
                "y": positions[1],
                "z": positions[2],
                "dish_dia": dish_diameters,
                "el_low": 2,
                "el_high": 90,
                "sefd": np.ones_like(stations, dtype=np.uint64()) * sefd
                if np.isscalar(sefd)
                else np.asarray(sefd),
                "altitude": np.ones_like(stations, dtype=np.uint64()) * altitude
                if np.isscalar(altitude)
                else np.asarray(altitude),
            }
        )

        return Layout.from_dataframe(df=df, rel_to_site=rel_to_site)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, rel_to_site: str | None = None):
        """Import a layout from a given pandas DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            DateFrame containing the layout.

        rel_to_site : str, optional
            The name of the site the coordinates are relative to.
            Is ignored is `None` or empty or `fmt`. Has to be an
            existing site for `astropy.coordinates.EarthLocation.of_site()`.
            Default: None
        """
        cls = cls()
        cls.names = df["station_name"]
        cls.cfg_path = "DataFrame"
        cls.rel_to_site = rel_to_site
        cls.x = df["x"]
        cls.y = df["y"]
        cls.z = df["z"]
        cls.dish_dia = df["dish_dia"]
        cls.el_low = df["el_low"]
        cls.el_high = df["el_high"]
        cls.sefd = df["sefd"]
        cls.altitude = df["altitude"]

        return cls

    @classmethod
    def from_url(cls, url: str, rel_to_site: str | None = None) -> "Layout":
        """Import a layout from a given URL.

        Parameters
        ----------
        url : str
            URL of the layout file.
        rel_to_site : str, optional
            The name of the site the coordinates are relative to.
            Is ignored if `None` or empty. Has to be an
            existing site for :func:`astropy.coordinates.EarthLocation.of_site()`
            Default: None
        """
        data = []
        for line in urllib.request.urlopen(url):
            data.append(line.decode("utf-8").split())

        df = pd.DataFrame(data)
        df = df.rename(columns=df.iloc[0]).drop(df.index[0])
        df.columns = map(str.lower, df.columns)
        df = df.astype(
            {
                "station_name": str,
                "x": float,
                "y": float,
                "z": float,
                "dish_dia": float,
                "el_low": float,
                "el_high": float,
                "sefd": float,
                "altitude": float,
            }
        )

        cls = cls()
        cls.names = df["station_name"]
        cls.cfg_path = url
        cls.rel_to_site = rel_to_site
        cls.x = df["x"]
        cls.y = df["y"]
        cls.z = df["z"]
        cls.dish_dia = df["dish_dia"]
        cls.el_low = df["el_low"]
        cls.el_high = df["el_high"]
        cls.sefd = df["sefd"]
        cls.altitude = df["altitude"]

        return cls


def loc2itrf(cx, cy, cz, locx=0.0, locy=0.0, locz=0.0):
    """Returns the given points locx, locy, locz, which are
    relative to a common central point cx, cy, cz as the
    absolute points on the earth.

    Modified version of a CASAtasks script
    https://open-bitbucket.nrao.edu/projects/CASA/repos/casa6/browse/casa5/gcwrap/python/scripts/simutil.py

    Parameters
    ----------
    locx: array_like or float
        The x-coordinate in relative coordinates
    locy: array_like or float
        The y-coordinate in relative coordinates
    locz: array_like or float
        The z-coordinate in relative coordinates
    cx: float
        The center's x-coordinate in WGS84 coordinates
    cy: float
        The center's y-coordinate in WGS84 coordinates
    cz: float
        The center's z-coordinate in WGS84 coordinates
    """

    lon, lat, alt = geocentric2geodetic(cx, cy, cz)

    locx, locy, locz = map(np.array, (locx, locy, locz))
    # from Rob Reid;  need to generalize to use any datum...
    phi, lmbda = map(np.deg2rad, (lat, lon))
    sphi = np.sin(phi)
    a = 6378137.0  # WGS84 equatorial semimajor axis
    b = 6356752.3142  # WGS84 polar semimajor axis
    ae = np.arccos(b / a)
    N = a / np.sqrt(1.0 - (np.sin(ae) * sphi) ** 2)

    factor = (N + locz + alt) * np.cos(phi) - locy * sphi

    clmb = np.cos(lmbda)
    slmb = np.sin(lmbda)

    cx = factor * clmb - locx * slmb
    cy = factor * slmb + locx * clmb
    cz = (N * (b / a) ** 2 + locz + alt) * sphi + locy * np.cos(phi)

    return cx, cy, cz


def itrf2loc(x, y, z, cx, cy, cz):
    """Returns the relative position of given points
    x, y, z to a common central point cx, cy, cz on the earth.

    Modified version of a CASAtasks script
    https://open-bitbucket.nrao.edu/projects/CASA/repos/casa6/browse/casa5/gcwrap/python/scripts/simutil.py

    Parameters
    ----------
    x: array_like or float
        The x-coordinate in WGS84 coordinates
    y: array_like or float
        The y-coordinate in WGS84 coordinates
    z: array_like or float
        The z-coordinate in WGS84 coordinates
    cx: float
        The center's x-coordinate in WGS84 coordinates
    cy: float
        The center's y-coordinate in WGS84 coordinates
    cz: float
        The center's z-coordinate in WGS84 coordinates
    """

    clon, clat, h = geocentric2geodetic(cx, cy, cz)

    ccoslon = np.cos(clon)
    csinlon = np.sin(clon)
    csinlat = np.sin(clat)
    ccoslat = np.cos(clat)

    if isinstance(x, float):  # weak
        x = [x]
        y = [y]
        z = [z]
    n = x.__len__()
    lat = np.zeros(n)
    lon = np.zeros(n)
    el = np.zeros(n)

    # do like MsPlotConvert
    for i in range(n):
        # translate w/o rotating:
        xtrans = x[i] - cx
        ytrans = y[i] - cy
        ztrans = z[i] - cz
        # rotate
        lat[i] = (-csinlon * xtrans) + (ccoslon * ytrans)
        lon[i] = (
            (-csinlat * ccoslon * xtrans)
            - (csinlat * csinlon * ytrans)
            + ccoslat * ztrans
        )
        el[i] = (
            (ccoslat * ccoslon * xtrans)
            + (ccoslat * csinlon * ytrans)
            + csinlat * ztrans
        )

    return lat, lon, el


def geocentric2geodetic(x, y, z):
    """Returns given WGS84 coordinates (x,y,z) as geodetic
    coordinates (longitude [deg], latitude [deg], altitude [meter])

    Parameters
    ----------
    x: array_like or float
        The x-coordinate in WGS84 coordinates
    y: array_like or float
        The y-coordinate in WGS84 coordinates
    z: array_like or float
        The z-coordinate in WGS84 coordinates
    """

    if np.isscalar(x):
        lon, lat, alt = EarthLocation.from_geocentric(x, y, z, "m").to_geodetic()
        lon = lon.deg
        lat = lat.deg
        alt = alt.value
    else:
        lon, lat, alt = np.array([]), np.array([]), np.array([])
        for i in range(0, len(x)):
            loc = EarthLocation.from_geocentric(x[i], y[i], z[i], "m")
            lon = np.append(lon, loc.lon.deg)
            lat = np.append(lat, loc.lon.deg)
            alt = np.append(alt, loc.height.value)

    return lon, lat, alt


def geodetic2geocentric(lon, lat, alt):
    """Returns given geodetic coordinates (longitude, latitude, altitude)
    coordinates as (x,y,z) in meters

    Parameters
    ----------
    lon: array_like or float
        The longitude in geodetic coordinates
    lat: array_like or float
        The latitude in geodetic coordinates
    alt: array_like or float
        The altitude in geodetic coordinates
    """

    if np.isscalar(lon):
        x, y, z = EarthLocation.from_geodetic(
            lon=lon, lat=lat, height=alt
        ).to_geocentric()
        x = x.value
        y = y.value
        z = z.value
    else:
        x, y, z = np.array([]), np.array([]), np.array([])
        for i in range(0, len(lon)):
            loc = EarthLocation.from_geodetic(lon=lon[i], lat=lat[i], height=alt[i])
            x = np.append(x, loc.x.value)
            y = np.append(y, loc.y.value)
            z = np.append(z, loc.z.value)
    return x, y, z
