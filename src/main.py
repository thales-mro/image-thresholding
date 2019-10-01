import cv2
import matplotlib.pyplot as plt
import numpy as np
from thresholding import *

def open_image(name):
    """
    it makes calls for openCV functions for reading an image based on a name

    Keyword arguments:
    name -- the name of the image to be opened
    grayscale -- whether image is opened in grayscale or not
        False (default): image opened normally (with all 3 color channels)
        True: image opened in grayscale form
    """
    img_name = 'input/' + name  + '.pgm'
    return cv2.imread(img_name, cv2.IMREAD_UNCHANGED)

def save_image(name, image):
    """
    it makes calls for openCV function for saving an image based on a name (path)
    and the image itself

    Keyword arguments:
    name -- the name (path) of the image to be saved
    image -- the image itself (numpy array)
    """
    image_name = 'output/' + name + '.pgm'
    cv2.imwrite(image_name, image)

def save_histogram(hist, name):
    """
    it makes calls for matplotlib function for painting the histogram on canvas, then saving it

    Keyword arguments:
    name -- the name (path) of the histogram to be saved
    hist -- the histogram itself
    """
    plt.clf()
    plt.plot(hist, color='k')
    plt.savefig('output/' + name + '.png')


def main():
    """
    Entrypoint for the code of project 02 MO443/2s2019

    For every input image, it creates thresholded images with global and
    local methods, for different global thresholds and different window
    sizes for local methods
    """

    # for inserting other images, add tem to /input folder and list them here
    images = (
        'baboon',
        'fiducial',
        'monarch',
        'peppers',
        'retina',
        'sonnet',
        'wedge'
    )

    window_sizes = (
        3,
        9,
        15,
        33,
        99
    )

    global_thresholds = (
        50,
        128,
        200
    )

    global_th_out_folder = "global-thresholding/"

    for image_name in images:
        print(image_name, "image:")

        image = open_image(image_name)

        original_histogram = calculate_histogram(image)
        save_histogram(original_histogram, '-original-histogram-' + image_name)

        print("\tGlobal thresholding:")
        for gt in global_thresholds:
            result, histogram = global_thresholding(image, gt)
            save_image(global_th_out_folder + str(gt) + '-threshold/' + image_name, result)
            save_histogram(histogram, global_th_out_folder +
                           str(gt) + '-threshold/' + image_name + 'histogram')

        print("\tLocal thresholding:")
        for ws in window_sizes:
            subfolder = str(ws) + 'x' + str(ws) + '-window/'

            result, histogram = bernsen_local_thresholding(image, ws)
            save_image(subfolder + image_name + '_bernsen', result)
            save_histogram(histogram, subfolder + image_name + '_bernsen_histogram')

            result, histogram = niblack_local_thresholding(image, ws)
            save_image(subfolder + image_name + '_niblack', result)
            save_histogram(histogram, subfolder + image_name + '_niblack_histogram')

            result, histogram = sauvola_pietaksinen_local_thresholding(image, ws)
            save_image(subfolder + image_name + '_sauvola-pietaksinen', result)
            save_histogram(histogram, subfolder + image_name + '_sauvola-pietaksinen_histogram')

            result, histogram = phansalskar_more_sabale_local_thresholding(image, ws)
            save_image(subfolder + image_name + '_phansalskar-more-sabale', result)
            save_histogram(histogram, subfolder + image_name + '_phansalskar-more-sabale_histogram')

            result, histogram = contrast_local_thresholding(image, ws)
            save_image(subfolder + image_name + '_contrast', result)
            save_histogram(histogram, subfolder + image_name + '_contrast_histogram')

            result, histogram = mean_local_thresholding(image, ws)
            save_image(subfolder + image_name + '_mean', result)
            save_histogram(histogram, subfolder + image_name + '_mean_histogram')

            result, histogram = median_local_thresholding(image, ws)
            save_image(subfolder + image_name + '_median', result)
            save_histogram(histogram, subfolder + image_name + '_median_histogram')

main()
