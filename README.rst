==================================
radiotools |ci| |codecov| |zenodo|
==================================

.. |ci| image:: https://github.com/radionets-project/radiotools/actions/workflows/ci.yml/badge.svg?branch=main
    :target: https://github.com/radionets-project/radiotools/actions/workflows/ci.yml?branch=main
    :alt: Test Status

.. |codecov| image:: https://codecov.io/github/radionets-project/radiotools/badge.svg
    :target: https://codecov.io/github/radionets-project/radiotools
    :alt: Code coverage

.. |zenodo| image:: https://zenodo.org/badge/807676503.svg
   :target: https://zenodo.org/badge/latestdoi/807676503
   :alt: Zenodo


Collection of tools for use in radio astronomy.


Installation
============

*radiotools* can be installed via `uv <https://docs.astral.sh/uv>`__ by calling

.. code::

  $ uv pip install radionets-radiotools

We recommend using a conda/mamba environment with python version ``>=3.11`` and ``<3.13``.

If you want to use features from the NRAO `CASAtools <https://pypi.org/project/casatools/>`_ package,
make sure you are using python 3.10 or 3.11.
