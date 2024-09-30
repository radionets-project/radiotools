"""Print information about radiotools and its cli tools."""

import importlib
import sys
from collections import OrderedDict
from importlib.metadata import distribution, requires
from textwrap import TextWrapper

import click
from packaging.requirements import Requirement

__all__ = ["main"]


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
    "--tools",
    is_flag=True,
    default=False,
    help="Print available cli tools",
)
def main(version, dependencies, tools):
    if len(sys.argv) <= 1:
        with click.get_current_context() as ctx:
            click.echo(ctx.get_help())
        sys.exit(1)

    if version:
        _info_version()

    if dependencies:
        _info_dependencies()

    if tools:
        _info_tools()


def _info_version():
    """Print version info."""
    import radiotools

    print("\n*** radiotools version info ***\n")
    print(f"version: {radiotools.__version__}")
    print("")


def _info_dependencies():
    """Print info about dependencies."""
    print("\n*** radiotools core dependencies ***\n")

    req = requires("radiotools")

    deps = [dep for dep in req if "dev" not in dep]

    for dep in deps:
        req = Requirement(dep)
        print(f"{req.name:>20} -- {str(req.specifier)}")


def _info_tools():
    """Print info about command line tools"""
    # Adapted from ctapipe
    # https://github.com/cta-observatory/ctapipe/blob/main/src/ctapipe/tools/info.py
    print("\n*** radiotools cli-tools ***\n")
    print("The following can be executed by typing radiotools-<toolname>:")
    print("")
    tools = {
        ep.name: ep.value
        for ep in distribution("radiotools").entry_points
        if ep.group in "console_scripts"
    }

    description = OrderedDict()
    for name, val in tools.items():
        module_name, attr = val.split(":")
        module = importlib.import_module(module_name)
        if hasattr(module, "__doc__") and module.__doc__ is not None:
            try:
                descr = module.__doc__
                descr.replace("\n", "")
                description[name] = descr
            except Exception as e:
                description[name] = f"[Could not parse docstring: {e}]"
        else:
            description[name] = "[No Documentation. Please add a docstring]"

    wrapper = TextWrapper(width=80, subsequent_indent=" " * 35)

    for name, descr in sorted(description.items()):
        text = f"{name:<30s} -- {descr}"
        print(wrapper.fill(text))
        print("")
    print("")


if __name__ == "__main__":
    main()
