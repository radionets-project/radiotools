"""Tool to convert uvfits files into measurement sets.
Single files or all files inside a directory can be converted.
"""

import click

from pathlib import Path
from natsort import natsorted
import numpy as np
from radiotools.utils import uvfits2ms, rmtree
import subprocess
from tqdm.auto import tqdm


@click.command()
@click.argument("fits_path", type=click.Path(exists=True, dir_okay=True))
@click.argument("ms_path", type=click.Path(exists=True, dir_okay=True))
@click.option("--log", default=False)
def main(fits_path, ms_path, log=False):
    """Convert uvfits files into measurement sets.

    Parameters
    ----------
    fits_path : str
        Path to fits file or directory with fits files
    ms_path : str
        Path to store ms file or directory for ms files
    """
    fits_path = Path(fits_path)
    ms_path = Path(ms_path)

    if ".ms" not in ms_path.name:
        ms_path.mkdir(parents=True, exist_ok=True)

    if fits_path.is_dir():
        fits_files = natsorted(np.array([x for x in fits_path.iterdir()]))
    else:
        fits_files = fits_path

    for i, path in enumerate(tqdm(fits_files)):
        if fits_path.is_dir():
            file_name = "vis_" + str(i) + ".ms"
            out = ms_path / file_name
        else:
            out = ms_path.parent + "/vis_" + str(i) + ".ms"
        if out.exists():
            rmtree(out)
        uvfits2ms(path, out)
        # if not log:
        #     subprocess.run("rm casa-*.log", shell=True)


if __name__ == "__main__":
    main()
