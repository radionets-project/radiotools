"""Tool to calculate and show source visibility for a given time,
location/array layout.
"""

import datetime

import click


def main():
    target = click.prompt(
        "Target [ICRS name, leave empty to enter RA/Dec]",
        type=str,
        default="",
        show_default=False,
    )
    from matplotlib.pyplot import show

    from radiotools.visibility.visibility import SourceVisibility

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
        target=target,
        date=date,
        location=location,
        obs_length=obs_length,
        print_optimal_date=True,
    )

    plot = click.confirm("Plot visibility?", default=False)
    if plot:
        fig, _ = vis.plot()
        show(block=True)


if __name__ == "__main__":
    main()
