_prefixes = {
    "microarcsecond": (3.6e9, "μas"),
    "µas": (3.6e9, "μas"),
    "milliarcsecond": (3.6e6, "mas"),
    "marcsec": (3.6e6, "mas"),
    "masec": (3.6e6, "mas"),
    "mas": (3.6e6, "mas"),
    "arcsecond": (3.6e3, "as"),
    "arcsec": (3.6e3, "as"),
    "asec": (3.6e3, "as"),
    "as": (3.6e3, "as"),
}

def px2radec(header, xcutout=None, ycutout=None, num_ticks=9, unit="mas", ax=None):
    if not unit in _prefixes.keys():
        raise ValueError(
            f"Unknown unit! Please provide one of {_prefixes.keys()}"
        )

    px_incr = header["CDELT2"]
    ref_pos = header["CRPIX1"] - 1, header["CRPIX2"] - 1
    naxis = header["NAXIS1"], header["NAXIS2"]

    if xcutout:
        xlim = ref_pos[0] - xclip[0], ref_pos[0] + xclip[1]
    else:
        xlim = (0, naxis[0])
    if ycutout:
        ylim = ref_pos[1] - yclip[0], ref_pos[1] + yclip[1]
    else:
        ylim = (0, naxis[1])

    if not isinstance(num_ticks, tuple):
        num_ticks = num_ticks, num_ticks

    xticks = np.linspace(*xlim, num_ticks[0], dtype=int)
    yticks = np.linspace(*ylim, num_ticks[1], dtype=int)

    xticklabels = xticks - ref_pos[0]
    yticklabels = yticks - ref_pos[1]

    xshift = xticklabels[xticklabels > 0][0]
    yshift = yticklabels[yticklabels > 0][0]

    xticks -= xshift.astype(int)
    yticks -= yshift.astype(int)

    xticklabels -= xshift
    yticklabels -= yshift
    xticklabels = [f"{label * px_incr * _prefixes[unit][0]:.2f}" for label in xticklabels]
    yticklabels = [f"{label * px_incr * _prefixes[unit][0]:.2f}" for label in yticklabels]

    if ax:
        ax.set(
            xlim=xlim,
            ylim=ylim + yshift,
            xticks=xticks,
            yticks=yticks,
            xticklabels=xticklabels,
            yticklabels=yticklabels,
            xlabel = rf"Relative RA $/\; \mathrm{{{_prefixes[unit][1]}}}$",
            ylabel = rf"Relative Dec $/\; \mathrm{{{_prefixes[unit][1]}}}$",
        )
    return xlim, ylim, xticks, yticks, xticklabels, yticklabels
