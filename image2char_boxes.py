from typing import List
import numpy as np
import cv2
import imutils
from PIL import Image as im
from scipy.ndimage import rotate
import matplotlib.pyplot as plt


def find_score(arr, angle):
    data = rotate(arr, angle, reshape=False, order=0)
    hist = np.sum(data, axis=1)
    score = np.sum((hist[1:] - hist[:-1]) ** 2)
    return hist, score


def rotate_img(img: np.ndarray, limit: int = 5, delta: int = 1):
    angles = np.arange(-limit, limit + delta, delta)
    best_score, best_angle, best_hist = 0, 0, None
    for angle in angles:
        hist, score = find_score(img, angle)
        if score > best_score:
            best_score = score
            best_angle = angle
            best_hist = hist
    
    print('Best angle: {}'.format(best_angle))
    
    return rotate(img, best_angle, reshape=False, order=0), best_hist


def preprocess_img(img: np.ndarray) -> np.ndarray:
    # Resize image to limit processing and for consistency
    img = imutils.resize(img, width=1080)
    
    # Denoising
    denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)
    
    # Make image a binary image
    img_grey = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    img_threshold = 255 - cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    return img_threshold


def get_line_imgs_from_img(img: np.ndarray) -> List[np.ndarray]:
    img = imutils.resize(img, width=1080)
    # img_1 = img.copy()
    # img_2 = img.copy()
    img = preprocess_img(img)
    img, hist = rotate_img(img)
    
    # Display image
    on_line = False
    line_start = 0
    line_imgs = []
    for i, val in enumerate(hist):
        if val > 510:
            if not on_line:
                on_line = True
                line_start = i
            # cv2.line(img_2, (0, i), (img_2.shape[0], i), (0, 255, 0), 1)
        elif on_line:
            on_line = False
            # cv2.imshow(f'{(line_start, i)}', img[line_start: i])
            line_imgs.append(img[line_start: i])
            
    """result = cv2.addWeighted(img_2, .5, img_1, .5, 0)
    print(y_ranges)
    cv2.imshow('1', result)
    cv2.waitKey(0)"""
    
    # max_val = np.max(hist)
    # min_val = np.min(hist)
    # print(hist.shape)
    # plt.plot(hist)
    # plt.show()
    
    return line_imgs


def get_char_boxes_in_line(line_img: np.ndarray) -> list:
    line_img, _ = rotate_img(line_img)
    hist = np.sum(line_img, axis=0)
    cv2.imshow('1', line_img)
    on_line = False
    line_start = 0
    char_imgs = []
    for i, val in enumerate(hist):
        if val > 0:
            if not on_line:
                if i - line_start > 3:
                    print(f"Space between {line_start} and {i}.")
                on_line = True
                line_start = i
            elif val == 255:
                if i - line_start > 4:
                    cv2.imshow(f'{(line_start, i)}', line_img[:, line_start: i])
                    char_imgs.append(line_img[:, line_start: i])
                line_start = i
            # cv2.line(img_2, (0, i), (img_2.shape[0], i), (0, 255, 0), 1)
        elif on_line:
            on_line = False
            if i - line_start > 4:
                cv2.imshow(f'{(line_start, i)}', line_img[:, line_start: i])
                char_imgs.append(line_img[:, line_start: i])
            line_start = i
    
    cv2.waitKey(0)



img_1 = cv2.imread('data/1.jpg')
line_lst = get_line_imgs_from_img(img_1)
get_char_boxes_in_line(line_lst[20])
"""img_1 = imutils.resize(img_1, width=1080)
rotated_img, hist_1 = rotate_img(img_1)
plt.plot(hist_1)
plt.show()"""
# cv2.imshow('1', img_1)
# cv2.imshow('2', rotated_img)
# cv2.waitKey(0)
