import sys
from argparse import ArgumentParser

__all__ = ["info"]

_dependencies = sorted(
    [
        "astropy >=6.1.0",
        "matplotlib ~=3.0",
        "numpy ~=1.16",
        "pyvisgen",
        "matplotlib",
    ]
)


def main(args=None):
    parser = ArgumentParser()

    parser.add_argument("--version", action="store_true", help="Print version number")
    parser.add_argument("--visibility", action="store_true")
    parser.add_argument("--dependencies", action="store_true")

    if len(sys.argv) <= 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args(args)
    info(**vars(args))


def info(version=False, visibility=False, dependencies=False):
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

    target = input("Target (either coordinates [RA/Dec] or an ICRS name): ")
    date = input("Date (YYYY-MM-DD HH:MM:SS): ")
    location = input("Location (either an array layout or an existing location): ")

    vis = SourceVisibility(target=target, date=date, location=location)
    fig, _ = vis.plot()

    show(block=True)


if __name__ == "__main__":
    main()
