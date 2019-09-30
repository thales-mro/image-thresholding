import cv2
import numpy as np


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
    return result

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
            #print(np.min(window) + np.max(window))
            local_threshold = (int(np.min(window)) + int(np.max(window)))//2
            if img[j][i] < local_threshold:
                result[j][i] = 255
    return result

def niblack_local_thresholding(img, window_size=3, k=1):
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
    return result