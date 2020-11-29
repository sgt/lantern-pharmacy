import os
import sys
from typing import Callable

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import color, io, feature, measure, filters


def load_image(name: str, directory: str = 'data') -> np.ndarray:
    filename = os.path.join(directory, name)
    if not filename.endswith('.jpg'):
        filename += '.jpg'
    return io.imread(filename)


def preprocess(img: np.ndarray) -> np.ndarray:
    return color.gray2rgb(img) if len(img.shape) == 2 else img


def attempt1_process(img: np.ndarray) -> np.ndarray:
    """
    Cleans up image with text.

    See https://github.com/andrewdcampbell/OpenCV-Document-Scanner

    requires higher resolution
    with current quality actually makes recognition less accurate
    do not use
    """
    # blurred = gaussian(image, sigma=3, truncate=0.00000001)
    # blurred = cv2.GaussianBlur(image, (0,0), 3)
    # sharpen = cv2.GaussianBlur(img, (0, 0), 3)
    sharpen = img
    sharpen = cv2.addWeighted(img, 1.5, sharpen, -0.5, 0)
    thresh = cv2.adaptiveThreshold(sharpen[:, :, 0], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15)
    return color.gray2rgb(thresh)


def show_side_by_side(img: np.ndarray, process_func: Callable[[np.ndarray], np.ndarray]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes.ravel()

    # tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
    # h, theta, d = hough_line(source, theta=tested_angles)

    # for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
    #     y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
    #     ax[2].plot(origin, (y0, y1), '-r')

    ax[0].imshow(img)
    ax[0].set_title('Source')
    ax[0].set_axis_off()

    processed = process_func(image)

    ax[1].imshow(processed)
    ax[1].set_title('Result')
    ax[1].set_axis_off()

    plt.show()


def show_corners(img: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(15, 6))

    # tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
    # h, theta, d = hough_line(source, theta=tested_angles)

    # for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
    #     y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
    #     ax[2].plot(origin, (y0, y1), '-r')

    ax.imshow(img, cmap=plt.cm.gray)
    ax.set_axis_off()

    coords = feature.corner_peaks(feature.corner_harris(img), min_distance=5, threshold_rel=0.4)
    coords_subpix = feature.corner_subpix(img, coords, window_size=13)
    ax.plot(coords[:, 1], coords[:, 0], color='cyan', marker='o', linestyle='None', markersize=6)
    ax.plot(coords_subpix[:, 1], coords_subpix[:, 0], '+r', markersize=15)

    plt.show()


def detect_paragraphs_orig(img: np.ndarray) -> None:
    """
    https://stackoverflow.com/questions/57249273/how-to-detect-paragraphs-in-a-text-document-image-for-a-non-consistent-text-stru
    """
    # Load image, grayscale, Gaussian blur, Otsu's threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Create rectangular structuring element and dilate
    # 4,4 2 - selects all columns except leftmost
    # 3,3 3 - ditto
    # 2,2 6 - ditto
    # 2,5 3 / 3,5 2 - works well with crisp, doesn't work with pale
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=4)

    # Find contours and draw rectangle
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)

    cv2.imshow('thresh', thresh)
    cv2.imshow('dilate', dilate)
    cv2.imshow('image', image)
    cv2.waitKey()


def detect_lines_cv2(img: np.ndarray) -> None:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    fig, ax = plt.subplots(figsize=(15, 6))

    for rho, theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(img, (x1, y1), (x2, y2), (255,0,0), 5)
        # plt.axline((x1, y1), (x2, y2), color='red')

    ax.imshow(img, cmap=plt.cm.gray)
    ax.set_axis_off()

    plt.show()


def detect_lines_skimage(img:np.ndarray)->None:
    fig, ax = plt.subplots(figsize=(15, 6))

    ax.imshow(img, cmap=plt.cm.gray)
    ax.set_axis_off()

    # plot detected lines

    plt.show()


if __name__ == '__main__':
    CRISP_IMG = ('crisp', '../test/data')
    PALE_IMG = ('pale', '../test/data')
    PALE_SMALL_IMG = ('pale-small', '../test/data')
    CRISP_SMALL_IMG = ('crisp-small', '../test/data')

    source = load_image(*CRISP_IMG)
    # source = load_image(*CRISP_SMALL_IMG)
    image = preprocess(source)
    # show_side_by_side(image, attempt1_process)
    # show_corners(image[:, :, 0])
    # show_contours(image[:, :, 0])
    # imsave("../test/data/crisp-small-proc.jpg", result)

    # detect_paragraphs_orig(image)
    detect_lines_cv2(image)
