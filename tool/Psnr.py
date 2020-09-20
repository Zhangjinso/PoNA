import math
from skimage.io import imread, imsave


from PIL import Image


def get_image_data(filename):
    """
    This gets the width and height of an image (in pixels) and the total number of pixels of it.
    :param filename: image file
    :return: width, height, number of pixels
    """
    im = Image.open(filename)
    width = im.size[0]
    height = im.size[1]
    npix = im.size[0] * im.size[1]

    return width, height, npix


def get_rgb(filename, npix):
    """
    This gets the RGB values from a given file and saves them in three lists from a given file.
    :param filename: image file
    :param npix:  number of pixels
    :return: r, g, b
    """
    # Getting image pixels RGB values
    im = Image.open(filename)
    rgb_im = im.convert('RGB')

    # Creating three lists of npix items
    r = [-1] * npix
    g = [-1] * npix
    b = [-1] * npix

    for y in range(0, im.size[1]):
        for x in range(0, im.size[0]):
            # We get the RGB value in each pixel and save each component in an array
            rpix, gpix, bpix = rgb_im.getpixel((x, y))
            r[im.size[0] * y + x] = rpix
            g[im.size[0] * y + x] = gpix
            b[im.size[0] * y + x] = bpix

    return r, g, b


def get_yuv(filename):
    """
    This gets the YCbCr values from a given file and saves them in three lists from a given file.
    :param filename: image file
    :return: y, u, v
    """
    # Getting image pixels RGB values
    im = Image.open(filename)
    im = im.convert('YCbCr')

    y = []
    u = []
    v = []

    for pix in list(im.getdata()):
        y.append(pix[0])
        u.append(pix[1])
        v.append(pix[2])

    return y, u, v

def rgb_to_yuv(r, g, b):  # in (0,255) range
    """
     This converts three lists (red, blue, green) in their equivalent YUV lists.
    :param r:
    :param g:
    :param b:
    :return: Y, Cb, Cr
    """
    # All of these lists have the same length
    y = [0] * len(r)
    cb = [0] * len(r)
    cr = [0] * len(r)

    # This is just the formula to get YUV from RGB.
    for i in range(0, len(r)):
        y[i] = int(0.299 * r[i] + 0.587 * g[i] + 0.114 * b[i])
        cb[i] = int(128 - 0.168736 * r[i] - 0.331364 * g[i] + 0.5 * b[i])
        cr[i] = int(128 + 0.5 * r[i] - 0.418688 * g[i] - 0.081312 * b[i])

    return y, cb, cr

def calculate_psnr(original_y, encoded_y, original_cb, encoded_cb, original_cr, encoded_cr, npix):
    """
    This calculates the Peak Signal to Noise Ratio (PSNR) of the encoded image, comparing it to the original one.
    :param original_y:
    :param encoded_y:
    :param original_cb:
    :param encoded_cb:
    :param original_cr:
    :param encoded_cr:
    :param npix:
    :return:
    """
    error_y = 0  # Summatory of y squared errors
    error_cb = 0  # Summatory of cr squared errors
    error_cr = 0  # Summatory of cb squared errors
    for i in range(0, len(original_y)):
        dif_y = abs(original_y[i] - encoded_y[i])  # Simple error between predicted and original luminance
        dif_cb = abs(original_cb[i] - encoded_cb[i])  # Simple error between predicted and original cb
        dif_cr = abs(original_cr[i] - encoded_cr[i])  # Simple error between predicted and original cr
        error_y += dif_y * dif_y  # We add y square to the summatory
        error_cb += dif_cb * dif_cb  # We add cb square to the summatory
        error_cr += dif_cr * dif_cr  # We add cr square to the summatory

    mse_y = float(error_y) / float(npix)  # And we get the mean squared error per pixel
    mse_cb = float(error_cb) / float(npix)  # And we get the mean squared error per pixel
    mse_cr = float(error_cr) / float(npix)  # And we get the mean squared error per pixel

    if mse_y != 0:
        # We use 255*255 because we're using 8 bits for every luminance value
        psnr_y = float(-10.0 * math.log(mse_y / (255 * 255), 10))
    else:
        psnr_y = 0

    if mse_cb != 0:
        # We use 255*255 because we're using 8 bits for every luminance value
        psnr_cb = float(-10.0 * math.log(mse_cb / (255 * 255), 10))
    else:
        psnr_cb = 0

    if mse_cr != 0:
        # We use 255*255 because we're using 8 bits for every luminance value
        psnr_cr = float(-10.0 * math.log(mse_cr / (255 * 255), 10))
    else:
        psnr_cr = 0

    print('Y = ', psnr_y, ' Cb = ', psnr_cb, ' Cr = ', psnr_cr)


import sys
from optparse import OptionParser



def print_usage():
    print('USAGE: {} ORIGINAL ENCODED'.format(sys.argv[0]))


if __name__ == '__main__':
    parser = OptionParser()
    (options, args) = parser.parse_args()

    if len(args) < 2:
        print_usage()

    original = args[0]
    encoded = args[1]

    original_width, original_height, original_npix = get_image_data(original)
    encoded_width, encoded_height, encoded_npix = get_image_data(original)

    if original_width != encoded_width or original_height != encoded_height or original_npix != encoded_npix:
        print("ERROR: Images should have same dimensions. \n")
        exit(1)

    # Getting YUV values
    original_y, original_cb, original_cr = get_yuv(original)
    encoded_y, encoded_cb, encoded_cr = get_yuv(encoded)

    calculate_psnr(original_y, encoded_y, original_cb, encoded_cb, original_cr, encoded_cr, original_npix)
