"""Main file of the Optical Character Recognition application."""
import sys

import cv2
import pytesseract

import pre_processing


def ocr(image_path):
    """Method to do the Optical Character Recognition.

    Args:
        image_path: path to a image.
    Returns:
        output: text output read by the ocr.
    """
    image = cv2.imread(image_path)
    if image is None:
        print('Image does not exist.')
        sys.exit()

    image = pre_processing.pre_processing(image)
    contours = pre_processing.find_countours(image)

    output = ''
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        output += pytesseract.image_to_string(pre_processing.deskewing(
            image[y:y + h, x:x + w]), lang='por') + ' '

    output += '\n'

    return output


if __name__ == '__main__':
    args = sys.argv

    if len(args) == 1:
        print('Example: python3 cv-test <image.jpg> <output.txt>(optional)')
        sys.exit()

    output = ocr(args[1])

    if len(args) > 2:
        with open(args[2], 'w') as file:
            file.write(output)
    else:
        print(output)
