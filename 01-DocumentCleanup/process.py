import io
import os
import cv2

import pytesseract
import numpy as np
import matplotlib.pylab as plt
from PIL import Image
from fnmatch import fnmatch
from wand.image import Image as wi

# FILES ------------------------------------------------------------------------------------------------------------

def get_files_from(sub_path) :
    root_path = '/home/berlanga/Desktop/nuveo/candidate-data/01-DocumentCleanup/'
    files = []

    for r, d, f in os.walk(root_path + sub_path) :
        for file in f :
            if '.png' in file :
                files.append(os.path.join(r, file))
    return files


# Step 01 ----------------------------------------------------------------------------------------------------------
# Aplicando redução de ruído

def denoising(img_path, source, target) :
    img = cv2.imread(img_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(img, None, 9, 13)

    plt.imshow(denoised, 'gray')
    cv2.imwrite(img_path.replace(source, target).replace('.png', '_.png'), denoised)

    return denoised


# Step 02 ----------------------------------------------------------------------------------------------------------
# Aplicando smoothening e binarização

def thresholding_smoothening(img_path, source, target) :
    img = cv2.imread(img_path, 0)

    filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 41)
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    result = cv2.bitwise_or(img, closing)

    plt.imshow(result, 'gray')
    cv2.imwrite(img_path.replace(source, target).replace('.png', '_.png'), result)

    return result


# Step 03 ----------------------------------------------------------------------------------------------------------
# Rotacionando o texto em X graus

def skewing(img_path, source, target) :
    image = cv2.imread(img_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)

    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45 :
        angle = -(90 + angle)
    else :
        angle = -angle

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags = cv2.INTER_CUBIC, borderMode = cv2.BORDER_REPLICATE)

    plt.imshow(rotated, 'gray')
    cv2.imwrite(img_path.replace(source, target).replace('.png', '_.png'), rotated)

    return rotated

# PROCESS ----------------------------------------------------------------------------------------------------------

source = 'noisy_data'
target = 'step_1'
files = sorted(get_files_from(source))
for file in files :
    print('Processando ' + target + ' a partir de [' + file + ']...')
    denoising(file, source, target)

source = 'step_1'
target = 'step_2'
files = sorted(get_files_from(source))
for file in files :
    print('Processando ' + target + ' a partir de [' + file + ']...')
    thresholding_smoothening(file, source, target)

source = 'step_2'
target = 'final_data'
files = sorted(get_files_from(source))
for file in files :
    print('Processando ' + target + ' a partir de [' + file + ']...')
    skewing(file, source, target)

