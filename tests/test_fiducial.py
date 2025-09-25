from radiotools.fiducial import Fiducial
import numpy as np
from pathlib import Path

from astropy.io.fits import PrimaryHDU, Header

class TestFiducial:
    def test_fiducial(self):

        expected_keys = {
            "source_name",
            "observatory",
            "img_size",
            "cell_size",
            "fov",
            "frequency",
            "bandwidth",
            "wavelength",
            "src_ra",
            "src_dec",
            "img_ra",
            "img_dec",
            "obs_date",
            "flux_unit",
            "beam",
        }

        expected_beam_keys = {
            "bmin",
            "bmaj",
            "bpa",
        }


        fiducial = Fiducial("./tests/data/2200+420.j.2006_04_05.i_0.1.fits")

        hdu = fiducial.get_hdu()
        assert isinstance(hdu, PrimaryHDU)
        assert isinstance(fiducial.get_header(), Header)
        assert isinstance(fiducial.get_header(hdu=hdu), Header)
        assert fiducial.get_metadata(hdu=hdu) == fiducial.get_metadata()

        metadata = fiducial.get_metadata(hdu=hdu)
        maximum_info = fiducial.get_maximum_info(hdu=hdu)
        assert expected_keys == set(metadata.keys())
        assert expected_beam_keys == set(metadata["beam"].keys())

        assert np.allclose(fiducial.get_image(), fiducial.get_image(hdu=hdu))

        image = fiducial.get_image(hdu=hdu)
        assert image.shape == (1024, 1024)
        assert np.allclose(fiducial.get_image(flux_unit="mJy/beam", hdu=hdu), fiducial.get_image(flux_unit="Jy/beam", hdu=hdu) * 1000)
        assert np.allclose(fiducial.get_image(flux_unit="mJy/pix", hdu=hdu), fiducial.get_image(flux_unit="Jy/pix", hdu=hdu) * 1000)
        assert np.allclose(fiducial.get_image(ra_incr_right=False, hdu=hdu), image[:, ::-1])
        fiducial.get_image("Jy/beam")

        fiducial_no_zero = fiducial.preprocess(clean=False, flux_unit="Jy/pix", overwrite=True)
        assert fiducial_no_zero._fits_path.exists()
        assert fiducial_no_zero.get_image().min() == 0
        assert fiducial_no_zero.get_metadata()["flux_unit"] == "Jy/pix"

        fiducial_cleaned = fiducial.preprocess(output_path="./tests/data/fits_no_zeros.fits", clean=True, crop=([400, 600], [500, 700]), overwrite=True)
        assert fiducial_cleaned._fits_path.exists()
        assert fiducial_cleaned.get_image().shape == (200, 200)

        cleaned_maximum_info = fiducial_cleaned.get_maximum_info()
        assert np.isclose(cleaned_maximum_info[0], maximum_info[0])
        assert np.allclose(cleaned_maximum_info[1].value, [112, 12])
        assert np.allclose(cleaned_maximum_info[2].value, maximum_info[2].value)

        fiducial_cleaned._fits_path.unlink(missing_ok=True)
        fiducial_no_zero._fits_path.unlink(missing_ok=True)


    def test_plot(self):
        fiducial = Fiducial("./tests/data/2200+420.j.2006_04_05.i_0.1.fits")
        fiducial.plot()
        fiducial.plot(display_beam=False)
        fiducial.plot(ax_unit="arcsec")
        fiducial.plot(ax_unit="arcsec", use_relative_ax=False)
        fiducial.plot(flux_unit="mJy/pix")
