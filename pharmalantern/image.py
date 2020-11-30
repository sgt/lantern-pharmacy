import os
from typing import Tuple, List

import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_grayscale_image(name: str, directory: str = 'data') -> np.ndarray:
    """
    Loads 2D bitmap from file.
    """
    filename = os.path.join(directory, name)
    if not filename.endswith('.jpg'):
        filename += '.jpg'
    rgb = cv2.imread(filename)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def plot_multiple(data: List[Tuple[np.ndarray, str, str]], layout=None) -> None:
    _layout = (1, len(data)) if layout is None else layout
    assert _layout[0] * _layout[1] >= len(data)

    fig, axes = plt.subplots(*_layout, figsize=(30, 20))
    ax = axes.ravel()

    for i, (img, title, cmap) in enumerate(data):
        if cmap is None:
            ax[i].imshow(img)
        else:
            ax[i].imshow(img, cmap=cmap)
        ax[i].set_title(title)
        ax[i].set_axis_off()

    plt.show()


def visualize_segmentation_process(gray: np.ndarray) -> None:
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # cv2.MORPH_ERODE - thin out thick stuff (prefer cv2.erode?)
    # cv2.MORPH_DILATE - thicken stuff (prefer cv2.dilate?)
    # cv2.MORPH_OPEN - remove white spots on background
    # cv2.MORPH_CLOSE - remove black spots on foreground

    # use morphology erode to blur horizontally
    # kernel = np.ones((500,3), np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 20))
    morph1 = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)
    morph2 = morph1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph2 = cv2.morphologyEx(morph1, cv2.MORPH_OPEN, kernel)

    # find contours
    cntrs = cv2.findContours(morph2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]

    # Draw contours
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    result = img.copy()
    for c in cntrs:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0), 2)

    plot_multiple([
        (img, 'Source', None),
        (thresh, 'Threshold', 'gray'),
        (morph1, 'Morph 1', 'gray'),
        (morph2, 'Morph 2', 'gray'),
        (result, 'Result', None),
    ], layout=(2, 3))
