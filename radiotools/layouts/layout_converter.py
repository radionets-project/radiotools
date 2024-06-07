from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import EarthLocation

pd.options.display.float_format = "{:f}".format


class Layout:

    """
    A tool to convert radio telescope array layout config files between different types.
    """

    def __init__(self):
        None

    def is_relative(self):
        return not (self.rel_to_site is None or self.rel_to_site == "")

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
            Is ignored is `None` or empty or `fmt` is not set to 'pyvisgen'.
            Has to be an existing site for `astropy.coordinates.EarthLocation.of_site()`.

        """

        FORMATS = ["casa", "pyvisgen"]

        file = Path(path)

        if file.exists():
            if overwrite:
                file.unlink()
            else:
                raise FileExistsError(
                    f"The file {file} already exists! If you want to overwrite it set overwrite=True!"
                )

        data = []

        match fmt:
            case "pyvisgen":
                data.append(
                    "station_name X Y Z dish_dia el_low el_high SEFD altitude\n"
                )

                for i in range(0, len(self.x)):
                    if not (rel_to_site is None or rel_to_site == ""):
                        location = EarthLocation.of_site(rel_to_site)

                        if self.is_relative():
                            prev_location = EarthLocation.of_site(self.rel_to_site)

                        row = map(
                            str,
                            [
                                self.names[i],
                                self.x[i] - location.x.value
                                if not self.is_relative()
                                else self.x[i]
                                - location.x.value
                                + prev_location.x.value,
                                self.y[i] - location.y.value
                                if not self.is_relative()
                                else self.y[i]
                                - location.y.value
                                + prev_location.y.value,
                                self.z[i] - location.z.value
                                if not self.is_relative()
                                else self.z[i]
                                - location.z.value
                                + prev_location.z.value,
                                self.dish_dia[i],
                                self.el_low[i],
                                self.el_high[i],
                                self.sefd[i],
                                self.altitude[i],
                            ],
                        )
                    else:
                        if self.is_relative():
                            prev_location = EarthLocation.of_site(self.rel_to_site)

                        row = map(
                            str,
                            [
                                self.names[i],
                                self.x[i]
                                if not self.is_relative()
                                else self.x[i] + prev_location.x.value,
                                self.y[i]
                                if not self.is_relative()
                                else self.y[i] + prev_location.y.value,
                                self.z[i]
                                if not self.is_relative()
                                else self.z[i] + prev_location.z.value,
                                self.dish_dia[i],
                                self.el_low[i],
                                self.el_high[i],
                                self.sefd[i],
                                self.altitude[i],
                            ],
                        )

                    data.append("\t".join(row) + "\n")

            case "casa":
                data.append("# X Y Z dish_dia station_name\n")

                if not (rel_to_site is None or rel_to_site == ""):
                    raise ValueError(
                        "You attempted to save relative coordinates to a NRAO CASA "
                        "layout, which is not possible because CASA uses absolute coordinates. "
                        "You have to set the rel_to_site parameter to None or empty str!"
                    )

                for i in range(0, len(self.x)):
                    if self.is_relative():
                        prev_location = EarthLocation.of_site(self.rel_to_site)

                    row = map(
                        str,
                        [
                            self.x[i]
                            if not self.is_relative()
                            else self.x[i] + prev_location.x.value,
                            self.y[i]
                            if not self.is_relative()
                            else self.y[i] + prev_location.y.value,
                            self.z[i]
                            if not self.is_relative()
                            else self.z[i] + prev_location.z.value,
                            self.dish_dia[i],
                            self.names[i],
                        ],
                    )
                    data.append(" ".join(row) + "\n")

            case _:
                raise ValueError(
                    f"{fmt} is not a valid format! Possible formats are: {', '.join(FORMATS)}!"
                )

        with open(file, "w", encoding="utf-8") as f:
            f.writelines(data)

    def __str__(self):
        return f"Configuration file loaded from: {self.cfg_path}\nRelative to site: {self.rel_to_site}\n{self.df}"

    def display(self):
        """
        Prints all information contained in the layout
        """
        print(self.__str__())

    def as_relative(self, rel_to_site):
        """
        Returns a copy of the current layout in relative coordinates.

        Parameters
        ----------
        rel_to_site : str
            The name of the site the coordinates are supposed to be relative to.
            Has to be an existing site for `astropy.coordinates.EarthLocation.of_site()`.

        """

        def gen_rng_file(increment=0):
            return Path(
                f"./temp_cfg_{np.random.default_rng().uniform(0,len(list(Path('.').glob('*'))) + increment)}.cfg"
            )

        temp_path = gen_rng_file()

        i = 1
        while temp_path.is_file():
            temp_path = gen_rng_file(i)
            i += 1

        self.save(temp_path, rel_to_site=rel_to_site)
        new_layout = Layout.from_pyvisgen(temp_path, rel_to_site=rel_to_site)

        temp_path.unlink()
        new_layout.cfg_path = self.cfg_path

        return new_layout

    def as_absolute(self):
        """
        Returns a copy of the current layout in absolute coordinates.
        Requires the layout to be in relative coordinates.
        """

        if not self.is_relative():
            raise TypeError(
                "This layout is not saved in relative coordinates and can therefore "
                "not be converted in absolute coordinates."
            )

        def gen_rng_file(increment=0):
            return Path(
                f"./temp_cfg_{np.random.default_rng().uniform(0,len(list(Path('.').glob('*'))) + increment)}.cfg"
            )

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

    def plot(self, save_to_file=""):
        """
        Generates a plot of the arrangement of the layout.

        Parameters
        ----------
        save_to_file : str, optional
            The name of the file the plot should be saved to.

        """

        if self.is_relative():
            layout = self.as_absolute()
        else:
            layout = self

        singular_alt = len(np.unique(layout.altitude)) == 1

        options = {
            "color": "#f54254" if singular_alt else None,
            "cmap": "cividis" if not singular_alt else None,
            "c": layout.altitude if not singular_alt else None,
        }

        fig, ax = plt.subplots(1, 1)

        im = ax.scatter(self.x, self.y, **options)

        if not singular_alt:
            fig.colorbar(im, ax=ax, label="Altitude")

        ax.set_xlabel("Geocentric x in m")
        ax.set_ylabel("Geocentric y in m")
        ax.set_title(f"Array Layout\n({layout.cfg_path.split('/')[-1]})")

        if save_to_file != "":
            fig.savefig(save_to_file)

        return fig, ax

    @classmethod
    def from_casa(cls, cfg_path, el_low=15, el_high=85, sefd=0, altitude=0):
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
        cls.rel_to_site = None
        cls.x = df.iloc[:, 0].to_list()
        cls.y = df.iloc[:, 1].to_list()
        cls.z = df.iloc[:, 2].to_list()
        cls.dish_dia = df.iloc[:, 3].to_list()
        cls.names = df.iloc[:, 4].to_list()
        cls.el_low = np.repeat(el_low, len(cls.x)) if np.isscalar(el_low) else el_low
        cls.el_high = (
            np.repeat(el_high, len(cls.x)) if np.isscalar(el_high) else el_high
        )
        cls.sefd = np.repeat(sefd, len(cls.x)) if np.isscalar(sefd) else sefd
        cls.altitude = (
            np.repeat(altitude, len(cls.x)) if np.isscalar(altitude) else altitude
        )

        df.insert(5, "el_low", cls.el_low)
        df.insert(6, "el_high", cls.el_high)
        df.insert(7, "sefd", cls.sefd)
        df.insert(8, "altitude", cls.altitude)

        cls.df = df

        return cls

    @classmethod
    def from_pyvisgen(cls, cfg_path, rel_to_site=None):
        """
        Import a layout from a radionets CASA layout config.

        Parameters
        ----------
        cfg_path : str
            The path of the config file to import.

        rel_to_site : str, optional
            The name of the site the coordinates are relative to.
            Is ignored is `None` or empty or `fmt`.
            Has to be an existing site for `astropy.coordinates.EarthLocation.of_site()`.
        """

        df = pd.read_csv(
            cfg_path,
            delimiter="\s+",
            encoding="utf-8",
            skip_blank_lines=True,
            skiprows=1,
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
        cls = cls()
        cls.names = df.iloc[:, 0].to_list()
        cls.cfg_path = cfg_path
        cls.rel_to_site = rel_to_site
        cls.x = df.iloc[:, 1].to_list()
        cls.y = df.iloc[:, 2].to_list()
        cls.z = df.iloc[:, 3].to_list()
        cls.dish_dia = df.iloc[:, 4].to_list()
        cls.el_low = df.iloc[:, 5].to_list()
        cls.el_high = df.iloc[:, 6].to_list()
        cls.sefd = df.iloc[:, 7].to_list()
        cls.altitude = df.iloc[:, 8].to_list()
        cls.df = df

        return cls
