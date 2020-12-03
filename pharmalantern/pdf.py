import logging
import os
from typing import Dict, Iterable

import numpy as np
import pikepdf
from pikepdf import PdfImage


def _get_pdf_image_page(pdf, page_num: int) -> PdfImage:
    page = pdf.pages[page_num]
    raw_image = list(page.images.values())[0]
    return PdfImage(raw_image)


def _get_pdf_image_pages(pdf, start: int, end: int) -> Iterable[PdfImage]:
    for num in range(start, end + 1):
        yield _get_pdf_image_page(pdf, num)


def get_page(pdf, page_num: int) -> np.ndarray:
    """
    Extract page as cv2 bitmap.
    """
    pdf_image = _get_pdf_image_page(pdf, page_num)
    pil_image = pdf_image.as_pil_image()
    img_arr = np.asarray(pil_image)
    return img_arr


def get_pages(pdf, start: int, end: int) -> Iterable[np.ndarray]:
    for num in range(start, end + 1):
        yield get_page(pdf, num)


def save_page(pdf, page_num: int, directory: str, filename: str) -> None:
    """
    Extract page then save it as jpg file.
    """
    pdf_image = _get_pdf_image_page(pdf, page_num)
    pdf_image.extract_to(fileprefix=os.path.join(directory, filename))


def save_selected_pages(pdf_filename: str, pages_dict: Dict[str, int]) -> None:
    """
    Extract and save pages of interest.
    """
    with pikepdf.open(pdf_filename) as pdf:
        for name, num in pages_dict.items():
            logging.info(f"extracting page {num} as {name}")
            save_page(pdf, num, 'test/data', name)
