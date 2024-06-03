import datetime
import sys

import click

__all__ = ["main"]

_dependencies = sorted(
    [
        "astropy >=6.1.0",
        "matplotlib ~=3.0",
        "numpy ~=1.16",
        "pyvisgen",
        "matplotlib",
    ]
)


@click.command()
@click.option(
    "--version",
    is_flag=True,
    show_default=True,
    default=False,
    help="Print version number",
)
@click.option(
    "--dependencies",
    is_flag=True,
    show_default=True,
    default=False,
    help="Print dependencies",
)
@click.option(
    "--visibility",
    is_flag=True,
    show_default=True,
    default=False,
    help="Show visibility",
)
def main(version, dependencies, visibility):
    if len(sys.argv) <= 1:
        with click.get_current_context() as ctx:
            click.echo(ctx.get_help())
        sys.exit(1)

    if version:
        _info_version()

    if dependencies:
        _info_dependencies()

    if visibility:
        _visibility()


def _info_version():
    """Print version info."""
    import radiotools

    print("\n*** radiotools version info ***\n")
    print(f"version: {radiotools.__version__}")
    print("")


def _info_dependencies():
    """Print info about dependencies."""
    print("\n*** radiotools core dependencies ***\n")

    for name in _dependencies:
        print(f"{name:>20s}")


def _visibility():
    from matplotlib.pyplot import show

    from radiotools.visibility import SourceVisibility

    target = click.prompt(
        "Target [ICRS name, leave empty to enter RA/Dec]",
        type=str,
        default="",
        show_default=False,
    )

    if target == "" or target.isspace():
        ra = click.prompt("Right ascension / deg", type=float)
        dec = click.prompt("Declination / deg", type=float)
        target = (ra, dec)

    date = click.prompt(
        "Date [Format: YYYY-MM-DD HH:MM:SS; leave empty for current time]",
        type=str,
        default="",
        show_default=False,
    )

    obs_length = click.prompt(
        "Observation length [hours]", type=float, default=8.0, show_default=True
    )

    if date == "" or date.isspace():
        date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    location = click.prompt("Location [array layout or an existing location]")

    vis = SourceVisibility(
        target=target, date=date, location=location, obs_length=obs_length
    )

    plot = click.confirm("Plot visibility?", default=False)
    if plot:
        fig, _ = vis.plot()
        show(block=True)


if __name__ == "__main__":
    main()
