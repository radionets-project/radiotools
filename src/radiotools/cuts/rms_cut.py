# This module was originally implemented in github.com/radionets-project/radio_stats
# by Christian Arauner (christian.arauner@tu-dortmund.de)

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def box_size(img: np.ndarray, amount_boxes: int) -> int:
    """Calculates the edge length of the calculation boxes.

    Parameters
    ----------
    img : np.ndarray, shape(N,M)
        Image to be analyzed.
    amount_boxes : int
        Amount of the boxes to be calculated.
        Needs to be quadratic number.

    Returns
    -------
    length : int
        Length of the boxes for the calculation.
    """
    if np.sqrt(amount_boxes).round(0) != np.sqrt(amount_boxes):
        raise ValueError("Amount boxes has to be a quadratic number!")
    return int(img.shape[0] / np.sqrt(amount_boxes))


def check_validity(
    img: np.ndarray, amount_boxes: int = 36, threshold: float = 1e-3
) -> np.ndarray:
    """Checks if the source is inside of the drawn boxes.
        If the source is inside the box it will be returned as False,
        otherwise as True.
        If every box cannot be used, the function recalls itself and
        doubles the threshold.

    Parameters
    ----------
    img : np.ndarray, shape(M,N)
        Image to be analyzed
    amount_boxes : int, default: 36
        Amount of the boxes to be calculated.
        Needs to be quadratic number.
    threshold : float
        Cutoff value to determine if the source is inside the box.

    Returns
    -------
    use_box : np.ndarray, shape(N,M)
        2d array of the usability of the boxes.
        If the box can be used, eg. no source in box, it will return True,
        otherwise False.
    """
    size = box_size(img, amount_boxes)
    row = int(np.sqrt(amount_boxes))
    use_box = np.zeros([row, row], dtype=bool)

    for j in range(row):
        for k in range(row):
            if np.any(
                img[size * j : size * (j + 1), size * k : size * (k + 1)] > threshold
            ):
                use_box[j, k] = False
            else:
                use_box[j, k] = True

    if np.sum(use_box) < row:
        use_box = check_validity(img, amount_boxes, threshold * 2)

    return use_box


def plot_boxes(img: np.ndarray, boxes: np.ndarray, axis=None, **kwargs):
    """Plots the boxes on top of the original image.

    Parameters
    ----------
    img : np.ndarray, shape(N,M)
        Image to be analyzed
    boxes : np.ndarray, shape(N,M)
        2d array boolean of the usability of the boxes.
    axis : `~matplotlib.axes.Axes`, optional
        Axes on whick the plot should be drawn.
        If None a new figure will be created.

    Returns
    -------
    plot : `~matplotlib.image.AxesImage`
        Plot of the original image colourd with the boxes.
        Red -> box not used, Green -> box used
    """
    row = boxes.shape[0]
    size = box_size(img, row**2)
    if axis is None:
        fig, axis = plt.subplots(1, 1)
    plot = axis.imshow(img, interpolation="none", **kwargs)
    color = np.zeros(boxes.shape, dtype=str)
    color[boxes] = "g"
    color[np.invert(boxes)] = "r"
    for j in range(row):
        for k in range(row):
            axis.fill_between(
                [size * j, size * (j + 1)],
                size * k,
                size * (k + 1),
                color=color[k, j],
                alpha=0.3,
            )
    return plot


def calc_rms_boxes(img: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Calculates the root mean square of the boxes.

    Parameters
    ----------
    img : np.ndarray, shape(N,M)
        Image to be analyzed
    boxes : np.ndarray, shape(N,M)
        2d array boolean of the usability of the boxes.

    Returns
    -------
    rms_boxes: 2d np.array
        Root mean square for each box.
        If the source is inside one box, the rms is set to -1.
    """
    row = boxes.shape[0]
    size = box_size(img, row**2)
    rms_boxes = -np.ones(boxes.shape)

    for j in range(boxes.shape[0]):
        for k in range(boxes.shape[1]):
            if not boxes[k, j]:
                continue
            else:
                rms_boxes[k, j] = np.sqrt(
                    np.mean(
                        img[k * size : (k + 1) * size, j * size : (j + 1) * size] ** 2
                    )
                )

    return rms_boxes


def rms_cut(img: np.ndarray, sigma: float = 3, verbose: bool = False, **kwargs):
    """Cuts an image using the rms.
        All values below the mean of the rms times sigma are set to zero.

    Parameters
    ----------
    img : np.ndarray, shape(N,M) or array of 2d arrays
        Images  to be analyzed
    sigma : float, default: 3.0
        Multiplier for the cut value.

    Reurns
    ------
    cut_img : np.ndarray, shape(N,M) or array of 2d arrays
        Cut images.
    """
    _img = np.copy(img)
    if len(_img.shape) == 2:
        ranges = calc_rms_boxes(_img, check_validity(_img, **kwargs))
        range_img = np.mean(ranges[ranges >= 0])

        _img[_img < sigma * range_img] = 0

        return _img
    else:
        cut_img = []
        if verbose:
            img = tqdm(img)

        for pic in img:
            ranges = calc_rms_boxes(pic, check_validity(pic, **kwargs))
            range_img = np.mean(ranges[ranges >= 0])

            pic[pic < sigma * range_img] = 0
            cut_img.append(pic)

        return np.array(cut_img)
