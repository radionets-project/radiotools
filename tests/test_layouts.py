def test_layouts():
    from radiotools.layouts import Layout

    BASE_URL = "https://raw.githubusercontent.com/radionets-project/"
    BASE_URL += "pyvisgen/refs/heads/main/resources/layouts/"

    layout = Layout.from_url(BASE_URL + "eht.txt")

    assert len(layout.names) == 8

    names = ["ALMA50", "SMTO", "LMT", "Hawaii8", "PV", "PdBI", "SPT", "GLT"]
    assert set(layout.names) == set(names)

    assert layout.x.dtype == "float64"
    assert layout.y.dtype == "float64"
    assert layout.z.dtype == "float64"

    x = [2225037.1851, -1828796.2, -768713.9637]
    y = [-5441199.162, -5054406.8, -5988541.7982]
    z = [-2479303.4629, 3427865.2, 2063275.9472]
    dish_dia = [84.700000, 10.000000, 50.000000]
    el_low = [15.000000, 15.000000, 15.000000]
    el_high = [85.000000, 85.000000, 85.000000]
    sefd = [110.0, 11900.0, 560.0]
    altitude = [5030.0, 3185.0, 4640.0]

    assert set(layout.x[:3]) == set(x)
    assert set(layout.y[:3]) == set(y)
    assert set(layout.z[:3]) == set(z)
    assert set(layout.dish_dia[:3]) == set(dish_dia)
    assert set(layout.el_low[:3]) == set(el_low)
    assert set(layout.el_high[:3]) == set(el_high)
    assert set(layout.sefd[:3]) == set(sefd)
    assert set(layout.altitude[:3]) == set(altitude)
