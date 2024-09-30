import requests
from bs4 import BeautifulSoup


def get_array_names(url: str) -> list[str]:
    """Fetches array names from a given URL
    leading to a directory.

    Parameters
    ----------
    url : str
        URL leading to a directory containing layout
        txt files.

    Returns
    -------
    layouts : list[str]
        List of available layouts.
    """
    r = requests.get(url)
    soup = BeautifulSoup(r.text, features="html.parser")

    a_tags = soup.find_all("a")

    layouts = []
    for i in a_tags:
        if ".txt" in str(i.get("href")):
            layouts.append(i.get("aria-label").split(".txt")[0])

    layouts = list(set(layouts))

    return layouts
