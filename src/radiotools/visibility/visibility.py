"""Shows the source visibility at a given location and time."""

import datetime
from collections import namedtuple
from typing import Literal

import astropy.units as u
import dateutil.parser
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import AltAz, BaseCoordinateFrame, EarthLocation, SkyCoord
from astropy.time import Time
from rich.console import Console
from rich.table import Table

from radiotools.layouts import Layout
from radiotools.utils import get_array_names

COLORS = [
    "#fd7f6f",
    "#7eb0d5",
    "#b2e061",
    "#bd7ebe",
    "#ffb55a",
    "#ffee65",
    "#beb9db",
    "#fdcce5",
    "#8bd3c7",
    "#b3d4ff",
]

PYVISGEN_LAYOUTS = "https://raw.githubusercontent.com/radionets-project/"
PYVISGEN_LAYOUTS += "pyvisgen/refs/heads/main/resources/layouts/"
PYVISGEN = "https://github.com/radionets-project/pyvisgen/tree/main/resources/layouts"

LOCATIONS = [
    "alma",
    "alma_dsharp",
    "dsa2000W",
    "dsa2000_31b",
    "eht",
    "meerkat",
    "vla",
    "vlba",
]


class SourceVisibility:
    """Plots the source visibility for a given location
    and time range.

    Parameters
    ----------
    target : tuple or str
        Tuple of RA/Dec coordinates or string of valid
        target name (ICRS).
    date : str or list[str]
        Date or start and end points of a range of dates.
    location : str or :class:`astropy.coordinates.EarthLocation`, optional
        Name of an existing array layout included in pyvisgen,
        a location, or :class:`astropy.coordinates.EarthLocation`
        object of an observatory or telescope. Default: ``None``
    obs_length: float, optional
        Observation length in hours. Default: ``None``
    frame : str or :class:`astropy.coordinates.BaseCoordinateFrame`, optional
        Type of coordinate frame the source sky coordinates
        should represent. Defaults: ``'icrs'``
    print_optimal_date : bool, optional
        Prints the optimal date to observe. Default: False
    """

    def __init__(
        self,
        target: tuple | str,
        date: str | list[str],
        location: str | EarthLocation = None,
        obs_length: float = 4.0,
        frame: str | BaseCoordinateFrame = "icrs",
        min_alt: float = 15.0,
        max_alt: float = 85.0,
        print_optimal_date: bool = False,
    ) -> None:
        """Initializes the class with source and observation information.

        Parameters
        ----------
        target : tuple or str
            Tuple of RA/Dec coordinates or string of valid
            target name (ICRS).
        date : str or list[str]
            Date or start and end points of a range of dates.
        location : str or :class:`astropy.coordinates.EarthLocation`, optional
            Name of an existing array layout included in pyvisgen,
            a location, or :class:`astropy.coordinates.EarthLocation` object of an
            observatory or telescope. Default: ``None``
        obs_length: float, optional
            Observation length in hours. Default: ``None``
        frame : str or :class:`astropy.coordinates.BaseCoordinateFrame`, optional
            Type of coordinate frame the source sky coordinates
            should represent. Defaults: 'icrs'
        min_alt : float, optional
            The minimum altitude of the source (in degrees) above the horizon to
            be determined as visible. Default: 15.0
        max_alt : float, optional
            The maximum altitude of the source (in degrees) above the horizon to
            be determined as visible. Default: 85.0
        print_optimal_date: bool, optional
            If `True` prints the optimal date for the observation. Default: ``False``
        """
        if isinstance(target, tuple) or (
            isinstance(target, np.ndarray) and target.size == 2
        ):
            self.ra = u.Quantity(target[0], unit=u.deg)
            self.dec = u.Quantity(target[1], unit=u.deg)
            self.source = SkyCoord(self.ra, self.dec, frame=frame)
            self.target_name = None

        elif isinstance(target, str):
            self.source = SkyCoord.from_name(target, frame=frame)
            self.ra = self.source.ra
            self.dec = self.source.dec
            self.target_name = target

        else:
            raise TypeError(
                "Please either provide a valid target name or a RA/Dec tuple!"
            )

        if isinstance(location, str) and (
            (location in get_array_names(PYVISGEN)) or (location in LOCATIONS)
        ):
            self.name = location
            self.array = Layout.from_url(PYVISGEN_LAYOUTS + location + ".txt")

            self.location = EarthLocation.from_geocentric(
                self.array.x, self.array.y, self.array.z, unit=u.m
            )

        elif isinstance(location, str):
            self.name = location
            self.location = EarthLocation.of_address(location)

        elif isinstance(location, EarthLocation):
            self.name = location
            self.location = location

        elif isinstance(location, tuple) and list(map(type, location)) == [
            str,
            pd.DataFrame,
        ]:
            self.name = location[0]
            self.location = EarthLocation.from_geocentric(
                location[1].X,
                location[1].Y,
                location[1].Z,
                unit=u.m,
            )

        else:
            raise ValueError("Please provide a valid location!")

        self.date = date
        self.obs_length = obs_length

        if min_alt > max_alt:
            raise ValueError(
                "Expected 'min_alt' to be smaller than 'max_alt', but got "
                f"min_alt = {min_alt} and max_alt = {max_alt}."
            )

        if min_alt < 0 or min_alt > 90 or max_alt < 0 or max_alt > 90:
            raise ValueError(
                "Expected 'min_alt' and 'max_alt' to be between 0 and 90 degrees."
            )

        self.min_alt = min_alt
        self.max_alt = max_alt

        self._get_dates()
        self._get_pos()

        self.get_optimal_date(print_optimal_date)

    def _get_dates(self) -> None:
        """Creates a date range from the dates given
        when initializing this class.
        """
        if isinstance(self.date, str):
            start_date = dateutil.parser.parse(self.date)
            end_date = start_date + datetime.timedelta(1)
        else:
            start_date = dateutil.parser.parse(self.date[0])
            end_date = dateutil.parser.parse(self.date[1])

        self.dates = pd.date_range(start_date, end_date, periods=1000).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        self.dates = pd.to_datetime(self.dates)

    def _get_pos(self) -> None:
        """Creates the sky coordinates of the source
        throughout the time range for a given frame
        (default is ICRS) and calculates the altitude
        and azimuth in the AltAz frame.
        """
        self.source_pos = dict()

        if self.location.size == 1:
            altaz_frame = AltAz(obstime=Time(self.dates), location=self.location)
            self.source_pos[0] = self.source.transform_to(altaz_frame)

        elif self.location.size > 10:
            altaz_frame = AltAz(obstime=Time(self.dates), location=self.location[0])
            self.source_pos[0] = self.source.transform_to(altaz_frame)

        else:
            for i, loc in enumerate(self.location):
                altaz_frame = AltAz(obstime=Time(self.dates), location=loc)
                self.source_pos[i] = self.source.transform_to(altaz_frame)

    def _plot_config(self, ax) -> None:
        """Settings for the plot.

        Parameters
        ----------
        ax : matplotlib.axes.Axes.axis
            Axis object of the current figure.
        """
        locale_label_strings = {
            "EN": dict(xlabel="Time / hours", ylabel="Altitude / deg"),
            "DE": dict(xlabel="Zeit / Stunden", ylabel="Höhenwinkel / Grad"),
        }

        locale_legend_strings = {
            "EN": dict(title="Telescope ID"),
            "DE": dict(title="Teleskop-Nummer"),
        }

        # axis ticks
        ax["A"].set_xticks(
            ax["A"].get_xticks(), ax["A"].get_xticklabels(), rotation=45, ha="right"
        )

        # further axis settings
        ax["A"].set(
            **locale_label_strings[self.plot_locale],
            ylim=(0, 90),
            title=self.name,
        )
        ax["A"].xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(ax["A"].xaxis.get_major_locator())
        )
        ax["B"].axis("off")

        # legend is plotted on axis "B"
        if self.legend:
            handles, labels = ax["A"].get_legend_handles_labels()
            ax["B"].legend(
                ncols=2,
                handles=handles,
                labels=labels,
                loc="upper left",
                handlelength=4,
                **locale_legend_strings[self.plot_locale],
            )

        # Text is drawn on axis "B"
        text_anchor = ax["B"].get_window_extent()

        locale_text = {
            "EN": "Solid lines indicate that the source\n"
            + "is visible. The visibility window is\n"
            + f"limited to a range between {self.min_alt} deg\n"
            + f"and {self.max_alt} deg.",
            "DE": "Durchgezogene Linien zeigen, dass die Quelle\n"
            + "sichtbar ist. Das Sichtbarkeitsfenster ist\n"
            + f"zwischen {self.min_alt}° und {self.max_alt}°"
            + "limitiert.",
        }

        locale_text_large_array = {
            "EN": " For array layouts with\n"
            + "more than 10 stations, only the first\n"
            + "station (ID 0) is shown.",
            "DE": " Für Anordnungen mit mehr als\n"
            + "10 Stationen wird nur die erste Station\n"
            + "(Index 0) gezeigt.",
        }

        if self.descr_text:
            text = locale_text[self.plot_locale]

            if self.location.size > 10:
                text += locale_text_large_array[self.plot_locale]

            ax["B"].annotate(
                text,
                (0.025, 0.025),
                xycoords=text_anchor,
                fontsize=12,
                va="bottom",
                ha="left",
                color="#41424C",
            )

        ax["B"].annotate(
            f"RA:\t{self.ra:.5f}\nDec:\t{self.dec:.5f}".expandtabs(),
            (0.025, 0.4),
            xycoords=text_anchor,
            fontsize=12,
            va="bottom",
            ha="left",
            fontfamily="monospace",
            color="#41424C",
        )
        if self.target_name:
            ax["B"].annotate(
                self.target_name,
                (0.025, 0.5),
                xycoords=text_anchor,
                fontsize=13,
                va="bottom",
                ha="left",
                fontfamily="monospace",
                color="#232023",
            )

    def plot(
        self,
        figsize: tuple[int, int] = (10, 5),
        colors: list = COLORS,
        *,
        descr_text: bool = True,
        legend: bool = True,
        locale: Literal["EN", "DE"] = "EN",
    ) -> tuple:
        """Plots the visibility of the source at the given
        time range. Also plots the positions of the sun and moon
        if set to ``True``.

        Parameters
        ----------
        figsize : tuple[int, int], optional
            Figure size. Width, height in inches.
            Default: (10, 5)
        colors : list, optional
            List of colors. If nothing provided a default
            list of colors is used. Default: None
        descr_text : bool, optional
            Whether to put a descriptive text next to the plot.
            Default: True
        legend : bool, optional
            If True, show legend in the plot. Default: True

        Returns
        -------
        tuple
            Figure and axis objects.
        """
        self.descr_text = descr_text
        self.legend = legend
        self.plot_locale = locale

        colors = iter(colors)

        fig, ax = plt.subplot_mosaic(
            "AB",
            layout="constrained",
            figsize=figsize,
            width_ratios=[3, 2],
        )

        for i, source_pos in enumerate(self.source_pos.values()):
            color = next(colors)

            mask = np.logical_and(
                source_pos.alt > self.min_alt * u.deg,
                self.max_alt * u.deg > source_pos.alt,
            )
            visible = source_pos.alt.copy()
            visible.value[~mask] = np.nan

            ax["A"].plot(
                self.dates,
                source_pos.alt,
                linestyle=":",
                color=color,
            )
            ax["A"].plot(
                self.dates,
                visible,
                lw=4,
                color=color,
                label=f"{i}",
            )

        ax["A"].plot()

        self._plot_config(ax)

        return fig, ax

    def _time_delta(self, r1, r2):
        latest_start = max(r1.start, r2.start)
        earliest_end = min(r1.end, r2.end)

        delta = earliest_end - latest_start

        return max(0, delta)

    def get_optimal_date(self, print_result: bool = False) -> list:
        """Computes the best date to observe the target source.
        Returns a list of three :class:`~pandas.Timestamp` where the
        first and last are the best date :math:`\pm` `obs_length / 2`.

        Parameters
        ----------
        print_result : bool, optional
            If `True`, also prints the result. Default: `False`

        Returns
        -------
        result : list
            List of :class:`~pandas.Timestamp`.
        """
        times = dict()
        t_range = namedtuple("t_range", ["start", "end"])

        for key, val in self.source_pos.items():
            maximum = np.max(val.alt)
            if maximum > self.max_alt * u.deg or maximum < self.min_alt * u.deg:
                continue
            idx_max = np.argmax(val.alt)
            delta = datetime.timedelta(hours=self.obs_length / 2)
            times[key] = [
                self.dates[idx_max] - delta,
                self.dates[idx_max],
                self.dates[idx_max] + delta,
            ]

        # Reindex times in case a key was skipped due to alt check.
        # This would be the case if, e.g. key 0 and 3  did not pass
        # the check resulting in times = {1: ..., 2: ..., 4: ...}.
        # This reindexing then creates times = {0: ..., 1: ..., 2: ...}.
        times = dict(enumerate(times.values()))

        dt = np.zeros([len(times), len(times)])
        for i, key_i in enumerate(times):
            for j, key_j in enumerate(times):
                r1 = t_range(
                    start=times[key_i][0].to_datetime64(),
                    end=times[key_i][-1].to_datetime64(),
                )
                r2 = t_range(
                    start=times[key_j][0].to_datetime64(),
                    end=times[key_j][-1].to_datetime64(),
                )

                dt[i, j] = self._time_delta(r1, r2)

        if dt.sum(axis=0).size == 0:
            raise ValueError(
                "The source is not visible with the chosen parameters. "
                "Radiotools was not able to determine the optimal "
                "observation date!"
            )

        result = times[int(np.argmax(dt.sum(axis=0)))]

        if print_result:
            print("")
            tab = Table(title="*** Best observation time ***")
            tab.add_column("Station ID", justify="right", style="cyan")
            tab.add_column("Obs. time start")
            tab.add_column("Obs. time midpoint")
            tab.add_column("Obs. time end")

            tab.add_row(
                f"{np.argmax(dt.sum(axis=0))}",
                result[0].strftime("%Y-%m-%d %H:%M:%S"),
                result[1].strftime("%Y-%m-%d %H:%M:%S"),
                result[2].strftime("%Y-%m-%d %H:%M:%S"),
            )
            console = Console()
            console.print(tab)
            print("")

        return result
