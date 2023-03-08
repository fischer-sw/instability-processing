import os
import sys
import json
import glob
import logging

from PIL import Image

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mlt
import cv2

# setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s")


def get_image_files(config, case):
    """
    Function that reads image names from folder
    """
    dat_path = os.path.join(*config["data_path"], "raw_cases", case)
    if os.path.exists(dat_path) == False:
        logging.warning(f"No data found for case {case} at {dat_path}")
        return []
    images = glob.glob("*.tiff", root_dir=dat_path)
    return images

def get_config():
    """
    Function that reads config from file
    """
    path = os.path.join(sys.path[0], "config.json")
    with open(path) as f:
        cfg = json.load(f)
    return cfg

def calc_case_ratio():
    """
    Function that calulates the case ratio for one case
    """
    config = get_config()
    for cas in list(config["cases"]):
        images = get_image_files(config, cas)
        images.sort()
    convert2png(config, images, cas)
    substract_background(config, cas, images)
    

def convert2png(config, images, case):
    """
    Function that converst .tiff images to .png images and stores them in png_cases folder for a case
    """
    for file in images:
        new_dir_path = os.path.join(*config["data_path"], "png_cases", case)
        png_ims = glob.glob("*.png", root_dir=new_dir_path)
        if png_ims != []:
            logging.info(f"Already created png images for case {case}")
            return
        else:
            new_img_path = os.path.join(new_dir_path, file.split(".")[0]+ ".png")
            img_path = os.path.join(*config["data_path"], "raw_cases", case, file)
            if os.path.exists(new_img_path):
                continue
            else:
                if os.path.exists(new_dir_path) is False:
                    os.makedirs(new_dir_path)
            im = Image.open(img_path)
            logging.info(f"Saved image at {new_img_path}")
            save_image(config, "png_cases", file, im, case)

def read_image(config, folder, filename, case):
    """
    Function that reads an image from directory
    """
    dir_path = os.path.join(*config["data_path"], folder, case)
    img_path = os.path.join(dir_path, filename.split(".")[0] + ".png")
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    return img

def save_image(config, folder, filename, img, case):
    """
    Function that saves images
    """
    dir_path = os.path.join(*config["data_path"], folder, case)
    if os.path.exists(dir_path) is False:
        os.makedirs(dir_path)
        logging.info(f"Creating dir {folder} for case {case}")
    img_path = os.path.join(*config["data_path"], folder, case, filename.split(".")[0] + ".png")
    mlt.imsave(img_path, img)

def substract_background(config, case, images):
    """
    Function that supstracts background from image
    """
    logging.info(f"Substracting background from images for case {case}")

    background = read_image(config, "png_cases", images[0], case)
    for img in images[1:]:
        new_img_path = os.path.join(*config["data_path"], "background_removed", case, img.split(".")[0] + ".png")
        if os.path.exists(new_img_path):
            continue
        tmp_img = read_image(config, "png_cases", img, case)
        img_diff = cv2.subtract(tmp_img, background)
        logging.info(f"Removed backgound for image {img}")
        save_image(config, "background_removed", img, img_diff, case)
    return img_diff

if __name__ == "__main__":
    calc_case_ratio()