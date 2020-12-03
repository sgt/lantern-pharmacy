import logging
import os
import time

import click
import pikepdf

from pharmalantern.image import save_segments
from pharmalantern.pdf import save_selected_pages, get_page
from pharmalantern.segmenters import ColumnSegmenter

DEFAULT_PDF_FILE = 'c:/z/spb-address-book.pdf'
START_PAGE = 770
END_PAGE = 1520

DEFAULT_OUTPUT_DIRECTORY = 'data/out'

SELECTED_PAGES = {
    'warped': 773,
    'crisp': 948,
    'pale': 1382,
    'new_letter': 1387,
    'curvy': 897  # separation lines are not quite straight
}


@click.group()
def cli():
    pass


@cli.command(help="for testing purposes")
@click.option('--file', type=str, default=DEFAULT_PDF_FILE)
def extract_test_pages(file: str) -> None:
    save_selected_pages(file, SELECTED_PAGES)


@cli.command(help="test column recognition")
@click.option('--file', type=str, default=DEFAULT_PDF_FILE)
@click.option('--start', type=int, default=START_PAGE)
@click.option('--end', type=int, default=END_PAGE)
def check(file: str, start: int, end: int) -> None:
    with pikepdf.open(file) as pdf:
        for page_num in range(start, end + 1):
            page = get_page(pdf, page_num)
            segmenter = ColumnSegmenter(page)
            columns = segmenter.detect_columns()
            if len(columns) not in [5, 10]:
                logging.warning(f"page {page_num} has {len(columns)} columns")
                segmenter.detect_columns(visualise=True)
                break


@cli.command()
@click.option('--file', type=str, default=DEFAULT_PDF_FILE)
@click.option('--start', type=int, default=START_PAGE)
@click.option('--end', type=int, default=END_PAGE)
@click.option('--out-dir', type=str, default=DEFAULT_OUTPUT_DIRECTORY)
def extract_columns(file: str, start: int, end: int, out_dir: str):
    start_time = time.time()
    with pikepdf.open(file) as pdf:
        for page_num in range(start, end + 1):
            page = get_page(pdf, page_num)
            segmenter = ColumnSegmenter(page)
            columns = segmenter.detect_columns()
            if len(columns) in [5, 10]:
                page_out_dir = os.path.join(out_dir, f"page-{page_num:04}")
                logging.info(f"saving {len(columns)} columns for page {page_num} to {page_out_dir}")
                save_segments(page, columns, page_out_dir)
            else:
                logging.warning(logging.warning(f"page {page_num} has {len(columns)} columns, not saving"))
    logging.info(f"{time.time() - start_time:.2f} seconds elapsed")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    cli()
