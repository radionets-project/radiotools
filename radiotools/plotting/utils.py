import numpy as np

_PREFIXES = dict.fromkeys(["microarcsecond", "µas"], (3.6e9, "µas"))
_PREFIXES.update(
    dict.fromkeys(["milliarcsecond", "marcsec", "masec", "mas"], (3.6e6, "mas"))
)
_PREFIXES.update(dict.fromkeys(["arcsecond", "arcsec", "asec", "as"], (3.6e3, "as")))


def px2radec(header, xlim=None, ylim=None, num_ticks=9, unit="mas", ax=None):
    if unit not in _PREFIXES.keys():
        raise ValueError(f"Unknown unit! Please provide one of {_PREFIXES.keys()}")

    px_incr = header["CDELT2"]
    ref_pos = header["CRPIX1"] - 1, header["CRPIX2"] - 1
    naxis = header["NAXIS1"], header["NAXIS2"]

    if xlim:
        xlim = ref_pos[0] - xlim[0], ref_pos[0] + xlim[1]
    else:
        xlim = (0, naxis[0])
    if ylim:
        ylim = ref_pos[1] - ylim[0], ref_pos[1] + ylim[1]
    else:
        ylim = (0, naxis[1])

    if not isinstance(num_ticks, tuple):
        num_ticks = num_ticks, num_ticks

    xticks = np.linspace(*xlim, num_ticks[0], dtype=int, endpoint=True)
    yticks = np.linspace(*ylim, num_ticks[1], dtype=int, endpoint=True)

    xticklabels = xticks - ref_pos[0]
    yticklabels = yticks - ref_pos[1]

    xshift = xticklabels[xticklabels >= 0][0]
    yshift = yticklabels[yticklabels >= 0][0]

    xticklabels -= xshift
    yticklabels -= yshift
    xticklabels = [
        f"{label * px_incr * _PREFIXES[unit][0]:.2f}" if label != 0 else "0.00"
        for label in xticklabels
    ]
    yticklabels = [
        f"{label * px_incr * _PREFIXES[unit][0]:.2f}" if label != 0 else "0.00"
        for label in yticklabels
    ]

    if ax:
        ax.set(
            xlim=xlim,
            ylim=ylim,
            xticks=xticks,
            yticks=yticks,
            xticklabels=xticklabels,
            yticklabels=yticklabels,
            xlabel=rf"Relative RA $/\; \mathrm{{{_PREFIXES[unit][1]}}}$",
            ylabel=rf"Relative Dec $/\; \mathrm{{{_PREFIXES[unit][1]}}}$",
        )
    return xlim, ylim, xticks, yticks, xticklabels, yticklabels
