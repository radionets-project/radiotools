# This module was originally implemented in github.com/radionets-project/radio_stats
# by Christian Arauner (christian.arauner@tu-dortmund.de)

import numpy as np


def truncate(img, excess=5, symmetrical=True):
    length = len(img.shape)

    if length == 2:
        img = [img]
    else:
        trunc_img_coll = []
    for pic in img:
        center = np.argwhere(np.amax(pic) == pic)[0]
        points = np.argwhere(pic)

        left = np.abs(center[1] - np.min(points[:, 1]))
        bottom = np.abs(center[0] - np.min(points[:, 0]))
        right = np.abs(center[1] - np.max(points[:, 1])) + 1
        top = np.abs(center[0] - np.max(points[:, 0])) + 1

        # Check for negative indices
        if np.any(
            np.array(
                [
                    center[0] - bottom - excess,
                    center[0] + top + excess,
                    center[1] - left - excess,
                    center[1] + right + excess,
                ]
            )
            < 0
        ):
            excess = 0

        if symmetrical:
            maximum = np.amax([left, top, right, bottom])
            trunc_img = np.copy(
                pic[
                    center[0] - maximum - excess : center[0] + maximum + excess,
                    center[1] - maximum - excess : center[1] + maximum + excess,
                ]
            )
        else:
            trunc_img = np.copy(
                pic[
                    center[0] - bottom - excess : center[0] + top + excess,
                    center[1] - left - excess : center[1] + right + excess,
                ]
            )

        if length == 2:
            return trunc_img
        else:
            trunc_img_coll.append(trunc_img)

    return trunc_img_coll


def orig_size(fit_img, orig_img):
    newpic = np.zeros(orig_img.shape)
    newpic[0 : fit_img.shape[0], 0 : fit_img.shape[1]] += fit_img

    orig_center = np.argwhere(np.amax(orig_img) == orig_img)[0]
    fit_center = np.argwhere(np.amax(fit_img) == fit_img)[0]

    to_roll = orig_center - fit_center

    newpic = np.roll(newpic, to_roll[0], axis=0)
    newpic = np.roll(newpic, to_roll[1], axis=1)

    return newpic
