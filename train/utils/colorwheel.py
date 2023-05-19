import cv2
import numpy as np


def make_color_wheel_image(img_width, img_height):
    """
    Creates a color wheel based image of given width and height
    Args:
        img_width (int):
        img_height (int):

    Returns:
        opencv image (numpy array): color wheel based image
    """
    hue = np.fromfunction(lambda i, j: (np.arctan2(i-img_height/2, img_width/2-j) + np.pi)*(180/np.pi)/2,
                          (img_height, img_width), dtype=np.float)
    saturation = np.ones((img_height, img_width)) * 255
    value = np.ones((img_height, img_width)) * 255
    hsl = np.dstack((hue, saturation, value))
    color_map = cv2.cvtColor(np.array(hsl, dtype=np.uint8), cv2.COLOR_HSV2BGR)
    return color_map

