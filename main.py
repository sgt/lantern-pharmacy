import click
import pikepdf
import logging

from pharmalantern.pdf import extract_page, SELECTED_PAGES

CONTEXT = {
    'pdf_file': 'c:/z/spb-address-book.pdf'
}


@click.group()
@click.option('--file', type=str)
@click.pass_context
def cli(ctx, file: str):
    ctx.ensure_object(dict)

    if file:
        ctx.obj['pdf_file'] = file


@cli.command(help="for testing purposes")
@click.pass_context
def extract_test_pages(ctx):
    with pikepdf.open(ctx.obj['pdf_file']) as pdf:
        for name, num in SELECTED_PAGES.items():
            logging.info(f"extracting page {num} as {name}")
            extract_page(pdf, num, 'test/data', name)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    cli(obj=CONTEXT)
