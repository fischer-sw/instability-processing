import os
import sys
import json
import glob
import logging
import random

from PIL import Image
from operator import itemgetter
# from tifffile import imread

import cv2 as cv
import skimage as ski
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(filename)s - %(levelname)s - %(funcName)s - %(message)s")


def get_cfg():
    """
    Function that reads config from .json file
    """
    path = os.path.join(sys.path[0], "conf.json")
    with open(path) as f:
        cfg = json.load(f)
    return cfg

def show_image(img, title):
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.savefig(f"test_{title}.png")

def show_circle_image(img, circles):
    title = "Image with circles"
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        cv.circle(img, (x, y), r, (0,255, 0), 3) #draws a red line with a line strength of 3 onto the found circles
        cv.circle(img, (x, y), 2, (255, 0, 0), 3) #draws the center of the circle
    plt.imshow(img)
    plt.title('Image Detected Bubbles')
    plt.axis('off')
    plt.savefig(f"test_{title}.png")

def convert_images(config, imgs):
    for file in imgs:
        new_dir_path = os.path.join(sys.path[0], "data_tmp")
        new_img_path = os.path.join(new_dir_path, file.split(".")[0]+ ".png")
        img_path = os.path.join(*config["data_path"], file)
        if os.path.exists(new_img_path):
            continue
        else:
            if os.path.exists(new_dir_path) == False:
                os.makedirs(new_dir_path)
        im = Image.open(img_path)
        logging.info(f"Saved image at {new_img_path}")
        im.save(new_img_path)

def read_image(config, img_filename):
    """
    Function that reads an image from directory
    """
    
    img_path = os.path.join(sys.path[0], "data_tmp", img_filename.split(".")[0] + ".png")
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    img_np = np.array(img) # image2nparray
    show_image(img, "Original Image")
    return img

def run_case():
    """
    Function that runs one case
    """    
    config = get_cfg()
    case = config['data_path'][-1]
    logging.info(f"Running detection for case {case}")
    imgs_files = get_images(config)
    convert_images(config, imgs_files)
    rand_imgs_files = get_rand_imgs(config, imgs_files) 
    tmp_test = rand_imgs_files[1]
    tmp_test = imgs_files[80]
    img_base = subtract_background(config, imgs_files, tmp_test)
    inv_img = invert_image(img_base, tmp_test)

    pars = {
        "u_limit" : 220,
        "u_value" : 255,
        "l_value" : 0,
    }

    bin_img = create_binary_img(inv_img, pars)
    hole_img = filling_holes(bin_img)
    tmp_circles = detect_circles(config, hole_img)

def run_multiple_cases(parameters):
    """
    Function that runs mutiple cases for a given set of parameters
    """
    pass

def get_rand_imgs(config, imgs):
    """
    Function that selects n random images from a set of images 
    """
    n_test_imgs = config["n_images"]
    logging.info(f"Selecting {n_test_imgs} images randomly from {len(imgs)} available images")
    rand_tmp_idx = []
    # choose random index
    for i in range(n_test_imgs):
        rand_tmp_idx.append(random.randrange(0, len(imgs)))
    rand_tmp_idx.sort()
    rand_imgs = list(itemgetter(*rand_tmp_idx)(imgs))
    logging.info(f"Chose {rand_imgs}")
    return rand_imgs

def get_images(config):
    """
    Function that reads images from directoy
    """
    path = os.path.join(sys.path[0], "data_tmp")
    imgs = glob.glob("*.png", root_dir=path)
    case = config['data_path'][-1]
    logging.info(f"Found {len(imgs)} images for case {case}")
    return imgs

def subtract_background(config, image_file_names, img):
    """
    Function that supstracts background from image
    """
    logging.info(f"Substracting background from img {img}")
    background = read_image(config, image_file_names[0])
    tmp_img = read_image(config, img)
    img_diff = cv.subtract(tmp_img, background)
    show_image(img_diff, "Background supstracted")
    return img_diff

def invert_image(img, file_name):
    """
    Function that inverts image
    """
    logging.info(f"Inverting image {file_name}")
    inv = ski.util.invert(img)
    # show_image(inv, "Inverted image")
    return inv

def filling_holes(img):
    # Filling the holes
    logging.info("Filling holes")
    h, w = img.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    
    img_floodfill = img.copy()
    cv.floodFill(img_floodfill, mask, (0, 0), 255)
    show_image(img_floodfill, "floodfilled image")
    img_floodfill_inv = cv.bitwise_not(img_floodfill)
    show_image(img_floodfill_inv, "inverted floodfilled image")
    img_out = img | img_floodfill_inv
    show_image(img_floodfill, "end floodfill")
    return img_out
    # return

def create_binary_img(img, params):
    """
    Function that creates binary image
    """
    logging.info("Creating binary image by threshold operation for image")
    logging.info(f"Lowest color value = {img.min()}")
    img = ski.util.img_as_ubyte(img)
    thresh_val = 254
    vals, vals_count = np.unique(img, return_counts=True)
    # fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
    # axs.hist(vals_count, bins=vals)
    # axs.savefig("values histo.png")
    _, img_th = cv.threshold(img, thresh_val, params["u_value"], cv.THRESH_BINARY)
    show_image(img_th, "Image after theshold")
    return img_th

def detect_circles(config, img):
    h, w = img.shape[:2]
    # Detect circles in the image
    circles = cv.HoughCircles(img,
        cv.HOUGH_GRADIENT_ALT,
        dp=1,
        minDist=2,
        param1=400,
        param2=0.15,
        minRadius=round(w/400),
        maxRadius=0
    )
    show_circle_image(img, circles)

if __name__ == "__main__":
    run_case()
