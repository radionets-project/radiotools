"""Shows the source visibility at a given location and time."""

import datetime
from collections import namedtuple

import astropy.units as u
import dateutil.parser
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from rich.console import Console
from rich.table import Table

from pyvisgen.layouts import layouts

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


class SourceVisibility:
    """Plots the source visibility for a given location
    and time range.
    """

    def __init__(
        self,
        target: tuple or str,
        date: str or list[str],
        location: str or EarthLocation = None,
        obs_length: float = 4.0,
        frame="icrs",
        print_optimal_date: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        target : tuple or str
            Tuple of RA/Dec coordinates or string of valid
            target name (ICRS).
        date : str or list[str]
            Date or start and end points of a range of dates.
        location : str or astropy.coordinates.EarthLocation
            Name of an existing array layout included in pyvisgen,
            a location, or astropy `EarthLocation` object of an
            observatory or telescope.
        obs_length: float
            Observation length in hours.
        frame : str, optional, default='icrs'
            Type of coordinate frame the source sky coordinates
            should represent. Defaults to ICRS.
        """
        if isinstance(target, tuple):
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

        if isinstance(location, str) and location in layouts.get_array_names():
            self.name = location
            self.array = layouts.get_array_layout(location)

            self.location = EarthLocation.from_geocentric(
                self.array.x * u.m, self.array.y * u.m, self.array.z * u.m
            )

        elif isinstance(location, str):
            self.name = location
            self.location = EarthLocation.of_address(location)

        elif isinstance(location, EarthLocation):
            self.name = location
            self.location = location

        else:
            raise ValueError("Please provide a valid location!")

        self.date = date
        self.obs_length = obs_length

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
        # axis ticks
        ax["A"].set_xticks(
            ax["A"].get_xticks(), ax["A"].get_xticklabels(), rotation=45, ha="right"
        )

        # further axis settings
        ax["A"].set(
            xlabel="UT Time / hours",
            ylabel="Altitude / deg",
            ylim=(0, 90),
            title=self.name,
        )
        ax["A"].xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(ax["A"].xaxis.get_major_locator())
        )
        ax["B"].axis("off")

        # legend is plotted on axis "B"
        handles, labels = ax["A"].get_legend_handles_labels()
        ax["B"].legend(
            ncols=2,
            handles=handles,
            labels=labels,
            loc="upper left",
            handlelength=4,
            title="Telescope ID",
        )

        # Text is drawn on axis "B"
        text_anchor = ax["B"].get_window_extent()

        text = "Solid lines indicate that the source\n"
        text += "is visible. The visibility window is\n"
        text += "limited to a range between 15 deg\n"
        text += "and 85 deg. For array layouts with\n"
        text += "more than 10 stations, only the first\n"
        text += "station (ID 0) is shown."

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

    def plot(self, figsize: tuple[int, int] = (10, 5), colors: list = None) -> tuple:
        """Plots the visibility of the source at the given
        time range. Also plots the positions of the sun and moon
        if set to `True`.

        Parameters
        ----------
        figsize : tuple[int, int], optional, default=(10,5)
            Figure size. Width, height in inches.
        colors : list, optional
            List of colors. If nothing provided a default
            list of colors is used.

        Returns
        -------
        tuple : Figure and axis objects.
        """
        if colors is None:
            colors = iter(COLORS)
        else:
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
                source_pos.alt > 15 * u.deg, 85 * u.deg > source_pos.alt
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

    def get_optimal_date(self, print_result=False):
        times = dict()
        t_range = namedtuple("t_range", ["start", "end"])

        for key, val in self.source_pos.items():
            maximum = np.max(val.alt)
            if maximum > 85 * u.deg or maximum < 15 * u.deg:
                continue
            idx_max = np.argmax(val.alt)
            delta = datetime.timedelta(hours=self.obs_length / 2)
            times[key] = [
                self.dates[idx_max] - delta,
                self.dates[idx_max],
                self.dates[idx_max] + delta,
            ]

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

        result = times[np.argmax(dt.sum(axis=0))]

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
