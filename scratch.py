# %%
import numpy as np
import pikepdf

from pharmalantern.image import load_grayscale_image, visualize_segmentation_process
from pharmalantern.pdf import get_page

# %%


CRISP_IMG = ('crisp', 'test/data')
PALE_IMG = ('pale', 'test/data')
PALE_SMALL_IMG = ('pale-small', 'test/data')
CRISP_SMALL_IMG = ('crisp-small', 'test/data')

source = load_grayscale_image(*PALE_IMG)
print(source.shape)
visualize_segmentation_process(source)

# %%
with pikepdf.open('c:/z/spb-address-book.pdf') as pdf:
    img1 = get_page(pdf, 948)
    img2 = load_grayscale_image(*CRISP_IMG)
    print(img2.shape)
    assert np.array_equal(img1, img2)
