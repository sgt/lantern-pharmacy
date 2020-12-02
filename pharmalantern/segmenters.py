from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from toolz import groupby


class TopBottomCropper:
    """
    - thresholding
    - morph-open to remove small elements (i.e. all text hopefully)
    """

    def __init__(self, img: np.ndarray):
        self.img = img
        self.img_h, self.img_w = img.shape

        # detection settings
        self.elem_min_h = 2
        self.elem_max_h = int(self.img_h * 0.1)
        self.elem_min_w = int(self.img_w * 0.5)
        self.elem_max_w = self.img_w
        self.top_y_boundary = int(self.img_h * 0.15)
        self.bottom_y_boundary = int(self.img_h * 0.85)

    @staticmethod
    def _step_thresholding(gray_img: np.ndarray) -> np.ndarray:
        _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return thresh

    @staticmethod
    def _step_morph_open(thresh_img: np.ndarray) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
        return cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel, iterations=3)

    def is_valid_element(self, rect: Tuple[int, int, int, int]) -> bool:
        x, _, w, h = rect
        return (self.elem_min_h <= h <= self.elem_max_h and
                self.elem_min_w <= w <= self.elem_max_w and
                (self.is_top_element(rect) or self.is_bottom_element(rect)))

    def is_top_element(self, rect: Tuple[int, int, int, int]) -> bool:
        _, y, _, h = rect
        return y + h <= self.top_y_boundary

    def is_bottom_element(self, rect: Tuple[int, int, int, int]) -> bool:
        _, y, _, _ = rect
        return y >= self.bottom_y_boundary

    def _visualise_process(self, img_step2: np.ndarray, rectangles,
                           top_boundary: int, bottom_boundary: int) -> None:
        """
        Plot: original, dilated, rectangles, crop boundaries.
        """
        fig, axes = plt.subplots(2, 2, figsize=(30, 20))
        ax = axes.ravel()

        ax[0].imshow(self.img, cmap='gray')
        ax[0].set_axis_off()

        ax[1].imshow(img_step2, cmap='gray')
        ax[1].set_axis_off()

        rect_img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2RGB)
        for x, y, w, h in rectangles:
            cv2.rectangle(rect_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        ax[2].imshow(rect_img)
        ax[2].set_axis_off()

        crop_img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2RGB)
        cv2.line(crop_img, (0, top_boundary), (self.img_w, top_boundary), (0, 0, 255), thickness=3)
        cv2.line(crop_img, (0, bottom_boundary), (self.img_w, bottom_boundary), (0, 0, 255), thickness=3)
        ax[3].imshow(crop_img)
        ax[3].set_axis_off()

        # img = cv2.line(self.img, (0, top), (self.img_w, top), (0, 0, 255), thickness=3)
        # img = cv2.line(img, (0, bottom), (self.img_w, bottom), (0, 0, 255), thickness=3)
        plt.show()

    def detect_crop_boundaries(self, visualise=False) -> Tuple[int, int]:
        img_step1 = self._step_thresholding(self.img)
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

        return top_boundary, bottom_boundary

    def crop(self) -> np.ndarray:
        """
        Return image cropped on top and bottom according to detected decorative elements.
        """
        top, bottom = self.detect_crop_boundaries()
        return self.img[top:bottom, :]


class ColumnSegmenter:
    """
    - crop a bit from top and bottom to remove decorative elements that might interfere
    - inverse-threshold the image
    - dilate the image, lumping together letters and rows of text
    - open-morph, making spaces between columns more pronounced

    """

    @staticmethod
    def _step_thresholding(img):
        _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return thresh
