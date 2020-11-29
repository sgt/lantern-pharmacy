import os

from pikepdf import PdfImage

SELECTED_PAGES = {
    'crisp': 948,
    'pale': 1382,
    'new_letter': 1387,
    'curvy': 897  # separation lines are not quite straight
}


def extract_page(pdf, page_num: int, directory: str, filename: str) -> None:
    """
    Extract page then save it as jpg file.
    """
    page = pdf.pages[page_num]
    raw_image = list(page.images.values())[0]
    pdf_image = PdfImage(raw_image)
    pdf_image.extract_to(fileprefix=os.path.join(directory, filename))
