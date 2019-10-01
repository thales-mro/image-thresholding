import cv2
import math
import numpy as np


def calculate_histogram(img):
    """
    It calculates histogram from the given image

    Keyword arguments:
    img -- the input image (numpy array)
    """
    return cv2.calcHist([img], [0], None, [256], [0, 256])

def global_thresholding(img, threshold=128):
    """
    It makes global thresholding by separating background from object
    pictures according to specific threshold for all pixels in image

    Keyword arguments:
    img -- the image itself (numpy array)
    threshold -- the threshold value (scale of 0 to 255)
    """
    if threshold < 0 or threshold > 255:
        print("It cannot apply threshold value to the image")
        return

    result = np.where(img < threshold, 255, 0)
    result = np.uint8(result)
    histogram = calculate_histogram(result)
    print("\t\tRate of black pixels for threshold value", threshold, ":",
          histogram[0][0]/(histogram[0][0] + histogram[255][0]))

    return result, histogram

def bernsen_local_thresholding(img, window_size=3):
    """
    It applies Bernsen method for local thresholding in the image

    keyword arguments:
    img -- the image itself (numpy array)
    window_size -- the window size for calculations (window is a squared matrix)
    """
    result = np.zeros_like(img)

    padded_img = np.pad(img, (window_size//2, window_size//2), 'constant')

    img_height, img_width = img.shape
    for j in range(img_height):
        for i in range(img_width):
            window = padded_img[j:j + window_size, i:i + window_size]
            local_threshold = (int(np.min(window)) + int(np.max(window)))//2
            if img[j][i] < local_threshold:
                result[j][i] = 255

    histogram = calculate_histogram(result)
    print("\t\tRate of black pixels for Bernsen method: ",
          histogram[0][0]/(histogram[0][0] + histogram[255][0]))

    return result, histogram

def niblack_local_thresholding(img, window_size=15, k=-0.2):
    """
    It applies Niblack method for local thresholding in the image

    Keyword arguments:
    img -- the image itself (numpy array)
    window_size -- the window size for calculations (window is a squared matrix)
    """
    result = np.zeros_like(img)

    padded_img = np.pad(img, (window_size//2, window_size//2), 'constant')

    img_height, img_width = img.shape
    for j in range(img_height):
        for i in range(img_width):
            window = padded_img[j:j + window_size, i:i + window_size]
            mean = np.mean(window)
            std_dev = np.std(window)
            local_threshold = int(mean + k*std_dev)
            if img[j][i] < local_threshold:
                result[j][i] = 255

    histogram = calculate_histogram(result)
    print("\t\tRate of black pixels for Niblack method: ",
          histogram[0][0]/(histogram[0][0] + histogram[255][0]))

    return result, histogram

def sauvola_pietaksinen_local_thresholding(img, window_size=3, k=0.5, r=128):
    """
    It applies Sauvola and Pietaksinen method for local thresholding in the image

    Keyword arguments:
    img -- the image itself (numpy array)
    window_size -- the window size for calculations (window is a squared matrix)
    k -- k parameter
    r -- R parameter
    """
    result = np.zeros_like(img)

    padded_img = np.pad(img, (window_size//2, window_size//2), 'constant')

    img_height, img_width = img.shape
    for j in range(img_height):
        for i in range(img_width):
            window = padded_img[j:j + window_size, i:i + window_size]
            mean = np.mean(window)
            std_dev = np.std(window)
            local_threshold = int(mean*(1 + k*((std_dev/r) - 1)))
            if img[j][i] < local_threshold:
                result[j][i] = 255

    histogram = calculate_histogram(result)
    print("\t\tRate of black pixels for Sauvola and Pietaksinen method: ",
          histogram[0][0]/(histogram[0][0] + histogram[255][0]))

    return result, histogram

def phansalskar_more_sabale_local_thresholding(img, window_size=3, k=0.25, r=0.5, p=2, q=10):
    """
    It applies Phansalskar, More and Sabale method for local thresholding in the image

    Keyword arguments:
    img -- the image itself (numpy array)
    window_size -- the window size for calculations (window is a squared matrix)
    k -- k parameter
    r -- R parameter
    p -- p parameter
    q -- q parameter
    """
    result = np.zeros_like(img)

    padded_img = np.pad(img, (window_size//2, window_size//2), 'constant')

    img_height, img_width = img.shape
    for j in range(img_height):
        for i in range(img_width):
            window = padded_img[j:j + window_size, i:i + window_size]
            mean = np.mean(window)
            std_dev = np.std(window)
            local_threshold = int(mean*(1 + (p*math.exp(-q*mean)) + (k*(((std_dev)/r) - 1))))
            if img[j][i] < local_threshold:
                result[j][i] = 255

    histogram = calculate_histogram(result)
    print("\t\tRate of black pixels for Phansalskar, More and Sabale method: ",
          histogram[0][0]/(histogram[0][0] + histogram[255][0]))

    return result, histogram


def contrast_local_thresholding(img, window_size=3):
    """
    It applies Contrast method for local thresholding in the image

    Keyword arguments:
    img -- the image itself (numpy array)
    window_size -- the window size for calculations (window is a squared matrix)
    """
    result = np.zeros_like(img)

    padded_img = np.pad(img, (window_size//2, window_size//2), 'constant')

    img_height, img_width = img.shape
    for j in range(img_height):
        for i in range(img_width):
            window = padded_img[j:j + window_size, i:i + window_size]
            min_v = np.min(window)
            max_v = np.max(window)
            dist_min = abs(min_v - int(img[j][i]))
            dist_max = abs(max_v - int(img[j][i]))
            if dist_min < dist_max:
                result[j][i] = 255

    histogram = calculate_histogram(result)
    print("\t\tRate of black pixels for Contrast method: ",
          histogram[0][0]/(histogram[0][0] + histogram[255][0]))

    return result, histogram


def mean_local_thresholding(img, window_size=3):
    """
    It applies mean method for local thresholding in the image

    Keyword arguments:
    img -- the image itself (numpy array)
    window_size -- the window size for calculations (window is a squared matrix)
    """
    result = np.zeros_like(img)

    padded_img = np.pad(img, (window_size//2, window_size//2), 'constant')

    img_height, img_width = img.shape
    for j in range(img_height):
        for i in range(img_width):
            window = padded_img[j:j + window_size, i:i + window_size]
            mean = np.mean(window)
            if img[j][i] < mean:
                result[j][i] = 255

    histogram = calculate_histogram(result)
    print("\t\tRate of black pixels for Mean method: ",
          histogram[0][0]/(histogram[0][0] + histogram[255][0]))

    return result, histogram

def median_local_thresholding(img, window_size=3):
    """
    It applies median method for local htresholding in the image

    Keyword arguments:
    img -- the image itself (numpy array)
    window_size -- the window size for calculations (window is a squared matrix)
    """
    result = np.zeros_like(img)

    padded_img = np.pad(img, (window_size//2, window_size//2), 'constant')

    img_height, img_width = img.shape
    for j in range(img_height):
        for i in range(img_width):
            window = padded_img[j:j + window_size, i:i + window_size]
            median = np.median(window)
            if img[j][i] < median:
                result[j][i] = 255

    histogram = calculate_histogram(result)
    print("\t\tRate of black pixels for Median method: ",
          histogram[0][0]/(histogram[0][0] + histogram[255][0]))

    return result, histogram
