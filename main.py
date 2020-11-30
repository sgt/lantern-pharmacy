import logging

import click

from pharmalantern.pdf import save_selected_pages

CONTEXT = {
    'pdf_file': 'c:/z/spb-address-book.pdf'
}

SELECTED_PAGES = {
    'crisp': 948,
    'pale': 1382,
    'new_letter': 1387,
    'curvy': 897  # separation lines are not quite straight
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
    save_selected_pages(ctx.obj['pdf_file'], SELECTED_PAGES)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    cli(obj=CONTEXT)
