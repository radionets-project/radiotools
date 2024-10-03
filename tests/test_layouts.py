def test_layouts():
    from radiotools.layouts import Layout

    BASE_URL = "https://raw.githubusercontent.com/radionets-project/"
    BASE_URL += "pyvisgen/refs/heads/main/pyvisgen/layouts/"

    layout = Layout.from_url(BASE_URL + "eht.txt")
    assert len(layout.names) == 8
