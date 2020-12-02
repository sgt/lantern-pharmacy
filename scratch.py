# %%
import os

from pharmalantern.image import load_grayscale_image, visualise_segmentation_process, \
    visualise_decorative_elements_detection


# %%
from pharmalantern.segmenters import TopBottomCropper


def foo(x):
    return os.path.join('test/data', x + '.jpg')


CRISP_IMG = foo('crisp')
PALE_IMG = foo('pale')
PALE_SMALL_IMG = foo('pale-small')
CRISP_SMALL_IMG = foo('crisp-small')

source = load_grayscale_image(CRISP_IMG)

# %%
visualise_segmentation_process(source)

# %%
visualise_decorative_elements_detection(source)

# %%
cropper = TopBottomCropper(source)
cropper.detect_crop_boundaries(visualise=True)

