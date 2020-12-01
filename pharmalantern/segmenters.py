from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class TopBottomSegmenter:
    """
    - thresholding
    - morph-open to remove small elements (i.e. all text hopefully)
    """
    img: np.ndarray

    @staticmethod
    def _step_thresholding(gray_img: np.ndarray) -> np.ndarray:
        _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return thresh

    @staticmethod
    def _step_morph_open(thresh_img: np.ndarray) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
        return cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel, iterations=3)

    def visualise(self):
        pass  # todo not implemented

    def detect_rectangles(self) -> None:
        _, thresh = cv2.threshold(self.img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
        morph1 = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)

        contours, _ = cv2.findContours(morph1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [cv2.boundingRect(c) for c in contours]


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
