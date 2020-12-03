import logging
from typing import Tuple, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
from toolz import groupby

from pharmalantern.image import draw_rectangles, Rectangle


# todo rename back to cropper? introduce parameter - crop or erase.
# todo add a couple of pixels on top of top and bottom (3 seems to be enough)
class TopBottomEraser:
    """
    Detects unnecessary decorative elements (bars, ads) on top and bottom of the page
    and erases them, replacing with white background.
    """

    def __init__(self, img: np.ndarray, crop: bool = False, padding: int = 3):
        """
        :param img: image to process
        :param crop: if true, crops image, otherwise erases top and bottom with white rectangles
        :param padding: pixels to add to top and bottom in case there are artifacts left
        """
        self.img = img
        self.img_h, self.img_w = img.shape

        self.crop = crop
        self.padding = padding

        # detection settings
        self.elem_min_h = 2
        self.elem_max_h = int(self.img_h * 0.1)
        self.elem_min_w = int(self.img_w * 0.5)
        self.elem_max_w = self.img_w
        self.top_y_boundary = int(self.img_h * 0.15)
        self.bottom_y_boundary = int(self.img_h * 0.85)

    @staticmethod
    def _step_threshold(gray_img: np.ndarray) -> np.ndarray:
        _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return thresh

    @staticmethod
    def _step_morph_open(thresh_img: np.ndarray) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
        return cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel, iterations=3)

    def is_valid_element(self, rect: Rectangle) -> bool:
        x, _, w, h = rect
        return (self.elem_min_h <= h <= self.elem_max_h and
                self.elem_min_w <= w <= self.elem_max_w and
                (self.is_top_element(rect) or self.is_bottom_element(rect)))

    def is_top_element(self, rect: Rectangle) -> bool:
        _, y, _, h = rect
        return y + h <= self.top_y_boundary

    def is_bottom_element(self, rect: Rectangle) -> bool:
        _, y, _, _ = rect
        return y >= self.bottom_y_boundary

    def _visualise_process(self, img_step2: np.ndarray, rectangles,
                           top_boundary: int, bottom_boundary: int) -> None:
        """
        Plot: original, dilated, rectangles, erasure boundaries.
        """
        fig, axes = plt.subplots(2, 2, figsize=(30, 20))
        ax = axes.ravel()

        ax[0].imshow(self.img, cmap='gray')
        ax[0].set_axis_off()

        ax[1].imshow(img_step2, cmap='gray')
        ax[1].set_axis_off()

        rect_img = draw_rectangles(self.img, rectangles)
        ax[2].imshow(rect_img)
        ax[2].set_axis_off()

        # erasure_img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2RGB)
        # cv2.line(erasure_img, (0, top_boundary), (self.img_w, top_boundary), (0, 0, 255), thickness=3)
        # cv2.line(erasure_img, (0, bottom_boundary), (self.img_w, bottom_boundary), (0, 0, 255), thickness=3)
        # ax[3].imshow(erasure_img)

        cleaned_img = self._clean_up_image(self.img, top_boundary, bottom_boundary)
        ax[3].imshow(cleaned_img, cmap='gray')
        ax[3].set_axis_off()

        # ax[3].imshow(self.img[top_boundary:bottom_boundary, :], cmap='gray')

        # img = cv2.line(self.img, (0, top), (self.img_w, top), (0, 0, 255), thickness=3)
        # img = cv2.line(img, (0, bottom), (self.img_w, bottom), (0, 0, 255), thickness=3)
        plt.show()

    def detect_erasure_boundaries(self, visualise=False) -> Tuple[int, int]:
        img_step1 = self._step_threshold(self.img)
        img_step2 = self._step_morph_open(img_step1)

        contours, _ = cv2.findContours(img_step2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rectangles = [cv2.boundingRect(c) for c in contours]

        # filtering out elements that might be our objects of interest
        valid_elements = filter(self.is_valid_element, rectangles)
        grouped = groupby(self.is_top_element, valid_elements)
        top_elements = grouped.get(True, [])
        bottom_elements = grouped.get(False, [])

        top_boundary = max((y + h for _, y, _, h in top_elements), default=0)
        bottom_boundary = min((y for _, y, _, _ in bottom_elements), default=self.img_h)

        if visualise:
            self._visualise_process(img_step2, rectangles, top_boundary, bottom_boundary)

        return top_boundary + self.padding, bottom_boundary - self.padding

    @staticmethod
    def _clean_up_image(img: np.ndarray, top: int, bottom: int) -> np.ndarray:
        e = img.copy()
        h, w = img.shape
        cv2.rectangle(e, (0, 0), (w, top), (255, 255, 255), -1)
        cv2.rectangle(e, (0, bottom), (w, h), (255, 255, 255), -1)
        return e

    def cleaned_up_image(self) -> np.ndarray:
        """
        Return image with its top and bottom erased according to detected decorative elements.
        """
        top, bottom = self.detect_erasure_boundaries()

        # todo make cropping work
        return undefined if self.crop \
            else self._clean_up_image(self.img, top, bottom)


class ColumnSegmenter:
    """
    Extracting columns.
    """

    def __init__(self, img: np.ndarray):
        eraser = TopBottomEraser(img)
        self.img = eraser.cleaned_up_image()
        self.img_h, self.img_w = img.shape

        self.min_col_w = self.img_w * 0.15
        self.max_col_w = self.img_w * 0.22
        self.min_col_h = 100  # columns might be quite short on pages that switch over a letter

    @staticmethod
    def _step_threshold(gray_img: np.ndarray) -> np.ndarray:
        _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return thresh

    @staticmethod
    def _step_dilate(thresh: np.ndarray) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 17))
        return cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel, iterations=3)

    @staticmethod
    def _step_morph_open(dilated: np.ndarray) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
        return cv2.morphologyEx(dilated, cv2.MORPH_OPEN, kernel, iterations=3)

    def _visualise_process(self, img_step2: np.ndarray, img_step3: np.ndarray,
                           rectangles: List[Rectangle],
                           columns: List[Rectangle]) -> None:
        """
        Plot: original, dilated, morph, rectangles, columns.
        """
        fig, axes = plt.subplots(2, 3, figsize=(30, 20))
        ax = axes.ravel()

        ax[0].imshow(self.img, cmap='gray')
        ax[0].set_axis_off()

        ax[1].imshow(img_step2, cmap='gray')
        ax[1].set_axis_off()

        ax[2].imshow(img_step3, cmap='gray')
        ax[2].set_axis_off()

        rect_img = draw_rectangles(self.img, rectangles)
        ax[3].imshow(rect_img)
        ax[3].set_axis_off()

        columns_img = draw_rectangles(self.img, columns, color=(0, 255, 0))
        ax[4].imshow(columns_img)
        ax[4].set_axis_off()

        ax[5].set_axis_off()

        plt.show()

    def is_column(self, rect: Rectangle) -> bool:
        _, _, w, h = rect
        return (self.min_col_w <= w <= self.max_col_w) and self.min_col_h <= h

    def adjust_column(self, column: Rectangle, horizontal_padding=15) -> Rectangle:
        """
        Some heuristics for column adjustments.

        - apply horizontal padding
        - if column is leftmost or rightmost, extend it to the page's edge
        """
        x, y, w, h = column

        # horizontal padding
        x = max(x - horizontal_padding, 0)
        w = w + horizontal_padding * 2

        # adjusting leftmost or rightmost columns
        if x < self.min_col_w:
            x = 0
        elif x + w > self.img_w - self.min_col_w:
            w = self.img_w - x

        return x, y, w, h

    def detect_columns(self, visualise=False) -> List[Rectangle]:
        img_step1 = self._step_threshold(self.img)
        img_step2 = self._step_dilate(img_step1)
        img_step3 = self._step_morph_open(img_step2)

        contours, _ = cv2.findContours(img_step3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rectangles = [cv2.boundingRect(c) for c in contours]

        columns = list(filter(self.is_column, rectangles))

        if visualise:
            self._visualise_process(img_step2, img_step3, rectangles, columns)

        return [self.adjust_column(column) for column in columns]
