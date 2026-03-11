Radiotools v0.3.0 (2026-03-11)
==============================


API Changes
-----------

- Make print optional in ``radiotools.visibility.SourceVisibility`` class [`#1 <https://github.com/radionets-project/radiotools/pull/1>`__]

- - Add option to not show zeros in array layout plots in ``layouts.py``
  - Parameter restructuring in ``gridding.py`` and ``layouts.py`` [`#2 <https://github.com/radionets-project/radiotools/pull/2>`__]


Bug Fixes
---------

- - Temporary fix for inversion and rotation problem of gridder
  - Fix: Loading FITS in gridder was using wrong columns which did not
    allow other Stokes components than Stokes I to be loaded
  - Ungridded UV plot in Gridder uses correct values now and
    uses correct unit for axis [`#2 <https://github.com/radionets-project/radiotools/pull/2>`__]

- - Fix layout API of source visibility class [`#12 <https://github.com/radionets-project/radiotools/pull/12>`__]

- Fix bug in ``radiotools.cleaning.WSClean`` that handled ``mf-weighting`` flag incorrectly [`#14 <https://github.com/radionets-project/radiotools/pull/14>`__]

- Fix source vis class URL layout retrieval [`#24 <https://github.com/radionets-project/radiotools/pull/24>`__]


New Features
------------

- Add new submodule ``radiotools.cleaning`` and ``WSClean`` wrapper class [`#3 <https://github.com/radionets-project/radiotools/pull/3>`__]

- - Add new class method ``from_url`` to ``radiotools.layouts.Layout``
  - Add new utility function that fetches layout names from an URL via http-request [`#4 <https://github.com/radionets-project/radiotools/pull/4>`__]

- - Add analysis submodule
    - Add ``radiotools.analysis.get_source_rms`` function
    - Add ``radiotools.analysis.dynamic_range`` function that calculates the dynamic range as a function of the maximum pixel value of an image and the RMS of the source

  - Add ``rms`` function to ``radiotools.utils`` [`#8 <https://github.com/radionets-project/radiotools/pull/8>`__]

- - Add utility function `radiotools.utils.img2jansky` that converts an image in Jy/beam to Jy/px [`#9 <https://github.com/radionets-project/radiotools/pull/9>`__]

- - Add plotting submodule
  - Add ``radiotools.plotting.px2radec`` utility function [`#11 <https://github.com/radionets-project/radiotools/pull/11>`__]

- - Added `min_alt` and `max_alt` parameters to be able to adapt to simulation or telescope altitude restrictions. [`#15 <https://github.com/radionets-project/radiotools/pull/15>`__]

- - Added `desc_id` parameter to change what desc_id the gridded visibilities must have [`#16 <https://github.com/radionets-project/radiotools/pull/16>`__]

- - Add classmethod to generate an array layout from a CASA NRAO Measurement Set
  - Fix some docstrings
  - Add casadata initialization to CI workflow
  - Downgrade minimum python version to 3.10 [`#27 <https://github.com/radionets-project/radiotools/pull/27>`__]

- - Added the ``fiducial`` module to process and visualize cleaned radio images which are provided as FITS files.

  - Switched CASA table import in ``src/radiotools/layouts/layouts.py`` from ``casatools.table`` to ``casacore.table``, and updated the `table` instantiation to include ``ack=False`` for compatibility with ``python-casacore``.

  - Added ``python-casacore>=3.7.1`` to the main dependencies in `pyproject.toml`, and removed the optional CASA dependency group.

  - Replaced the ``img2jansky`` function with two new functions, ``beam2pix`` and ``pix2beam``, for converting between Jy/beam and Jy/pix units in ``src/radiotools/utils/utils.py``. Updated the corresponding exports in ``src/radiotools/utils/__init__.py``.

  - Removed the ``src/radiotools/measurements/measurements.py`` file and its corresponding import/export from ``src/radiotools/measurements/__init__.py``. [`#34 <https://github.com/radionets-project/radiotools/pull/34>`__]

- - Added German locale option to source visibility plot [`#36 <https://github.com/radionets-project/radiotools/pull/36>`__]


Maintenance
-----------

- Use towncrier for changelog generation [`#5 <https://github.com/radionets-project/radiotools/pull/5>`__]

- - Add simple tests for ``radiotools.visibility.SourceVisibility`` class and ``radiotools.layouts.Layout`` class
  - Add CI for tests [`#6 <https://github.com/radionets-project/radiotools/pull/6>`__]

- - Add Codecov report to CI [`#7 <https://github.com/radionets-project/radiotools/pull/7>`__]

- - Create docs with API reference
  - Update docstrings [`#10 <https://github.com/radionets-project/radiotools/pull/10>`__]

- - Remove forced rotation and inversion in plots
  - Change binning of grid
  - Use `numpy.float128` to ensure numerical stability
  - Add column name optional parameters to FITS import (default now `UU`, `VV`)
  - Remove flip and invert in `create_attributes` method [`#19 <https://github.com/radionets-project/radiotools/pull/19>`__]

- - Update maintainers of the package [`#21 <https://github.com/radionets-project/radiotools/pull/21>`__]

- - Made CASA optional
  - radiotools now uses hatch as build backend
  - Added more settings for ruff in ``pyproject.toml``
  - Fixed formatting throughout the codebase
  - Updated CI and added CD [`#22 <https://github.com/radionets-project/radiotools/pull/22>`__]

- Update docs with all new user and developer guides [`#26 <https://github.com/radionets-project/radiotools/pull/26>`__]

- - Removed the gridding module since it is now part of [``pyvisgrid``](https://github.com/radionets-project/pyvisgrid) [`#31 <https://github.com/radionets-project/radiotools/pull/31>`__]


Refactoring and Optimization
----------------------------

- - Refactor code in ``pyvisgen.layouts.Layout`` ``from_pyvisgen`` and
    ``from_casa`` class methods
  - Remove ``pyvisgen`` dependencies [`#4 <https://github.com/radionets-project/radiotools/pull/4>`__]

- - Add test for source visibility altitude restrictions in `tests/test_source_visibility.py`
  - Fix typo in `utils.py` [`#17 <https://github.com/radionets-project/radiotools/pull/17>`__]
