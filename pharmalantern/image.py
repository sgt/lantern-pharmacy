from typing import Tuple, List

import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_grayscale_image(filename: str) -> np.ndarray:
    """
    Loads 2D bitmap from file.
    """
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

    for i in range(layout[0] * layout[1]):
        ax[i].set_axis_off()

    plt.show()


def crop_vertical(img: np.ndarray, top: int = 0, bottom: int = 0) -> np.ndarray:
    h = img.shape[0]
    return img[top:h - bottom, :]


def draw_contours(img: np.ndarray, contours: list) -> np.ndarray:
    result = img.copy()
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0), 2)
    return result


def visualise_decorative_elements_detection(gray: np.ndarray) -> None:
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
    morph1 = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)

    contours, _ = cv2.findContours(morph1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = draw_contours(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB), contours)

    plot_multiple([
        (gray, 'Source', 'gray'),
        (thresh, 'Threshold', 'gray'),
        (morph1, 'Morph 1', 'gray'),
        (result, 'Result', 'gray'),
    ], layout=(2, 2))


def visualise_segmentation_process(gray: np.ndarray) -> None:
    """
    Experimental method for the segmentation pipeline. Plots every step.
    :param gray:
    :return:
    """
    # todo instead of crop, detect top and bottom horizontal elements
    gray = crop_vertical(gray, 80, 80)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # cv2.MORPH_ERODE - thin out thick stuff (prefer cv2.erode?)
    # cv2.MORPH_DILATE - thicken stuff (prefer cv2.dilate?)
    # cv2.MORPH_OPEN - remove white spots on background
    # cv2.MORPH_CLOSE - remove black spots on foreground

    # use morphology erode to blur horizontally
    # kernel = np.ones((500,3), np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 17))
    morph1 = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel, iterations=3)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
    morph2 = cv2.morphologyEx(morph1, cv2.MORPH_OPEN, kernel, iterations=3)

    contours, _ = cv2.findContours(morph2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = draw_contours(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB), contours)

    plot_multiple([
        (gray, 'Source', 'gray'),
        (thresh, 'Threshold', 'gray'),
        (morph1, 'Morph 1', 'gray'),
        (morph2, 'Morph 2', 'gray'),
        (result, 'Result', None),
    ], layout=(2, 3))
