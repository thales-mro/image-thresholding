import cv2
import numpy as np
from thresholding import *

def open_image(name, grayscale=False):
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

def main():
    """
    Entrypoint for the code of project 01 MO443/2s2019

    For every input image, it generates the colored and grayscale halftoning versions of images,
    varying the error propagation methods and the sweep order in image
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

    for image_name in images:
        print(image_name)
        image = open_image(image_name)
        result = global_thresholding(image)
        save_image(image_name + "_global-thresholding", result)
        result = bernsen_local_thresholding(image, 33)
        save_image(image_name + "_bernsen", result)
        result = niblack_local_thresholding(image)
        save_image(image_name + "_niblack", result)
        result = sauvola_pietaksinen_local_thresholding(image, 33)
        save_image(image_name + "_sauvola-pietaksinen", result)
        result = mean_local_thresholding(image, 33)
        save_image(image_name + "_mean", result)
        result = median_local_thresholding(image, 33)
        save_image(image_name + "_median", result)
main()
