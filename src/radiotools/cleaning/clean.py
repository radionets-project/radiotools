import subprocess as sp
from pathlib import Path

import toml


class WSClean:
    """Wrapper class for WSClean.

    Parameters
    ----------
    ms : str or Path
        Path to the measurement set.
    clean_config : dict or str or Path
        Either a dict containing the wsclean options or
        a path to a toml file containing the options.
        See note.
    create_skymodel : bool, optional
        Whether to create a skymodel before cleaning the image.
        Default: False.
    save_config : bool or str or Path, optional
        If True, save the config to a file relative to the file
        this class is called from. If given a path, the config
        is saved to that location instead. Default: False.

    Notes
    -----
    A valid config may look like the following:

    >>> clean_config = {
    >>>     'multiscale': True,
    >>>     'mgain': 0.5,
    >>>     'file_name': 'path/to/measurement_set',
    >>>     'data_column': 'DATA',
    >>>     'gain': 0.03,
    >>>     'weight': 'briggs 0',
    >>>     'mf_weighting': False,
    >>>     'size': [1024, 1024],
    >>>     'scale': '0.1masec',
    >>>     'pol': 'I',
    >>>     'niter': 5000000,
    >>>     'auto_threshold': 0.5,
    >>>     'auto_mask': 3,
    >>>     'padding': 1.3,
    >>>     'mem': 30,
    >>>     'verbose': True,
    >>> }
    """

    def __init__(
        self,
        ms: str | Path,
        clean_config: dict | str | Path = None,
        create_skymodel: bool = False,
        save_config: bool | str | Path = False,
    ) -> None:
        """Wrapper class for WSClean.

        Parameters
        ----------
        ms : str or Path
            Path to the measurement set.
        clean_config : dict or str or Path
            Either a dict containing the wsclean options or
            a path to a toml file containing the options.
            See note.
        create_skymodel : bool, optional
            Whether to create a skymodel before cleaning the image.
            Default: False.
        save_config : bool or str or Path, optional
            If True, save the config to a file relative to the file
            this class is called from. If given a path, the config
            is saved to that location instead. Default: False.
        """
        if not isinstance(clean_config, dict | str | Path):
            raise ValueError(
                "Please provide EITHER a dict object or a path"
                " to an existing configuration toml file."
            )

        if not Path(ms).is_dir():
            raise OSError(f"Measurement set {Path(ms).absolute()} does not exist.")

        self.ms = ms

        if isinstance(clean_config, str | Path):
            if not Path(clean_config).is_file():
                raise OSError(f"File {Path(clean_config).absolute()} does not exist.")

            with open(clean_config) as toml_file:
                clean_config = toml.load(toml_file)

        if save_config:
            self._save_config(clean_config, save_config)

        _clean_config = clean_config.copy()

        _clean_config["name"] = _clean_config.pop("file_name")
        _clean_config["data-column"] = _clean_config.pop("data_column")
        _clean_config["auto-threshold"] = _clean_config.pop("auto_threshold")
        _clean_config["auto-mask"] = _clean_config.pop("auto_mask")

        # We need to handle the case if size is an int and not a list.
        # This assumes, however, that x- and y- size are the same
        if isinstance(_clean_config["size"], int):
            _clean_config["size"] = [_clean_config["size"]] * 2

        # mf-weighting and no-mf-weighting are flags without arguments.
        # Since the wrapper only takes the bool mf_weighting as input,
        # we have to handle both cases and pass the respective WSClean
        # flags with an empty string as "argument".
        if _clean_config["mf_weighting"] is False:
            del _clean_config["mf_weighting"]
            _clean_config["no-mf-weighting"] = ""
        else:
            del _clean_config["mf_weighting"]
            _clean_config["mf-weighting"] = ""

        # Same as with mf-weighting, WSClean has two flags -- verbose
        # and quiet -- to handle output during cleaning. The wrapper
        # only takes the bool verbose as input, as such we have to
        # handle both cases.
        if _clean_config["verbose"] is False:
            del _clean_config["verbose"]
            _clean_config["quiet"] = ""
        else:
            del _clean_config["verbose"]

        # multiscale is another case of a flag without any arguments.
        # As such, we have to either delete the key and value pair,
        # or replace the bool with an empty string.
        if _clean_config["multiscale"] is False:
            del _clean_config["multiscale"]
        else:
            _clean_config["multiscale"] = ""

        self._clean_config = _clean_config

        if create_skymodel:
            self.skymodel_kwargs = _clean_config.copy()
            self.skymodel_kwargs["name"] += f"_{self.skymodel_kwargs['pol']}_skymodel"
            self.skymodel_kwargs["data-column"] = "DATA"
            self.skymodel_kwargs["auto-threshold"] = 1
            self.skymodel_kwargs["auto-mask"] = 3
            self.skymodel_kwargs["niter"] = 1000000

            self.create_skymodel()

    def create_skymodel(self) -> None:
        """Creates the skymodel using WSClean if ``create_skymodel``
        is set to ``True`` when initializing the class.
        """
        wsclean_opts = "wsclean "
        for key, val in self.skymodel_kwargs.items():
            if isinstance(val, list):
                wsclean_opts += f"-{key} {val[0]} {val[1]} "
            else:
                wsclean_opts += f"-{key} {val} "

        wsclean_opts += str(self.ms)
        sp.run(wsclean_opts, shell=True)

        print(f"Saved to {self.skymodel_kwargs['name']}<...>.fits")

    def clean_image(self) -> None:
        """Cleans the image using WSClean."""
        self._clean_config["name"] += "_" + self._clean_config["pol"]

        wsclean_opts = "wsclean "
        for key, val in self._clean_config.items():
            if isinstance(val, list):
                wsclean_opts += f"-{key} {val[0]} {val[1]} "
            else:
                wsclean_opts += f"-{key} {val} "

        wsclean_opts += self.ms
        sp.run(wsclean_opts, shell=True)

        print(f"Saved to {self._clean_config['name']}<...>.fits")

    def _save_config(self, _clean_config: dict, output_file: bool | str | Path) -> None:
        """Saves the config if ``save_config`` is set to ``True``
        when initializing the class.
        """
        if not isinstance(output_file, str | Path):
            output_file = Path(_clean_config["file_name"]).name
            output_file += f"_{_clean_config['pol']}" + "_config.toml"

        with open(output_file, "w") as toml_file:
            toml.dump(_clean_config, toml_file)
