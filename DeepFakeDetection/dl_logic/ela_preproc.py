from PIL import Image, ImageChops, ImageEnhance
import os

#NB: This wasn't use in the last version of the API.
def ela(input_image, quality):
    """
    Generates an ELA image from input image
    """
    tmp_fname = "temp_image"
    ela_fname = "result_image.jpeg"

    im = Image.fromarray(input_image)
    im = im.convert('RGB')
    im.save(tmp_fname, 'JPEG', quality=quality)

    tmp_fname_im = Image.open(tmp_fname)
    ela_im = ImageChops.difference(im, tmp_fname_im)

    extrema = ela_im.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0/max_diff
    ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)

    ela_im.save(ela_fname, 'JPEG')
    os.remove(tmp_fname)
    return ela_fname
