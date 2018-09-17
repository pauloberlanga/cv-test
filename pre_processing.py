"""File with the pre_processing methods to apply before ocr reading."""
import cv2
import numpy as np


def deskewing(image):
    """Method to rotate a skewed image.

    Args:
        image: skewed image.
    Returns:
        rotated: rotated image.
    """
    img = cv2.bitwise_not(image)
    thresh = cv2.threshold(img, 0, 255,
                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)

    return rotated


def find_countours(image):
    """Method to return contours of the image.

    Args:
        image: image to find the contours.
    Returns:
        contors: vector that represents the contours of the image.
    """
    kernel = np.ones((10, 5), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    contours = cv2.findContours(image, cv2.RETR_LIST,
                                cv2.CHAIN_APPROX_SIMPLE)[1]

    return contours


def pre_processing(image):
    """Method with the pre_processing steps.

    Args:
        image: raw image to be pre_processed.
    Returns:
        image: pre_processed image.
    """
    image = cv2.bilateralFilter(image, 3, 375, 75)

    if image.shape[0] < 1000:
        image = cv2.resize(image, None, fx=10, fy=10,
                           interpolation=cv2.INTER_CUBIC)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    return image
