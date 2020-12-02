# %%
import os

from pharmalantern.image import load_grayscale_image, visualise_segmentation_process
# %%
from pharmalantern.segmenters import ColumnSegmenter, TopBottomEraser


def foo(x):
    return os.path.join('test/data', x + '.jpg')


CRISP_IMG = foo('crisp')
PALE_IMG = foo('pale')
PALE_SMALL_IMG = foo('pale-small')
CRISP_SMALL_IMG = foo('crisp-small')
NEW_LETTER_IMG = foo('new_letter')
WARPED_IMG = foo('warped')

source = load_grayscale_image(WARPED_IMG)

# %%
visualise_segmentation_process(source)


# %%
cropper = TopBottomEraser(source)
cropper.detect_erasure_boundaries(visualise=True)
# cropper.cleaned_up_image()

#%%
segmenter = ColumnSegmenter(source)
segmenter.detect_columns(visualise=True)
