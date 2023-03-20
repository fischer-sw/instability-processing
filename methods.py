import os
import sys
import json
import glob
import logging
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mlt
import cv2

# setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s")


def get_image_files(config, case, folder):
    """
    Function that reads image names from folder
    """
    dir_path = os.path.join(*config["data_path"])
    if os.path.exists(dir_path) == False:
        logging.error(f"Data directory {dir_path} doesn't exsist. Please check config for valid path.")
        exit()

    dat_path = os.path.join(*config["data_path"], folder, case)
    if os.path.exists(dat_path) == False:
        logging.warning(f"No data found for case {case} at {dat_path}")
        return []
    images = glob.glob("*.tiff", root_dir=dat_path)
    if images == []:
        images = glob.glob("*.png", root_dir=dat_path)
    if images == []:
        logging.warning(f"No images found in folder {folder} for case {case}")
        tmp_images = []
    else:
        tmp_images = [x.split(".")[0] for x in images]

    if config["images"] != []:
        img_subset = []
        if not 0 in config["images"]:
            config["images"].append(0)
        for img in images:
            number = int(img.split("_")[0])
            if number in config["images"]:
                img_subset.append(img)
        tmp_images = img_subset

    tmp_images = [x.split(".")[0] for x in tmp_images]
    tmp_images.sort()
    return tmp_images

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
        # Reseting cases for easier development
        # reset_cases(config, cas)
        images = get_image_files(config, cas, "raw_cases")
        images.sort()
    convert2png(config, cas)
    make_histos(config, cas, "png_cases")
    substract_background(config, cas)
    make_histos(config, cas, "background_removed_cases")
    binarize_image(config, cas)
    make_histos(config, cas, "binary_cases")
    # create_animation(config, cas, "png_cases")
    # create_animation(config, cas, "binary_cases")

def convert2png(config, case):
    """
    Function that converst .tiff images to .png images and stores them in png_cases folder for a case
    """
    images = get_image_files(config, case, "raw_cases")

    for file in images:
        new_dir_path = os.path.join(*config["data_path"], "png_cases", case)
        new_img_path = os.path.join(new_dir_path, file + ".png")
        img_path = os.path.join(*config["data_path"], "raw_cases", case, file)
        if os.path.exists(new_img_path):
            continue
        else:
            if os.path.exists(new_dir_path) is False:
                os.makedirs(new_dir_path)
        # im = Image.open(img_path + ".tiff")
        im = cv2.imread(img_path + ".tiff", cv2.IMREAD_GRAYSCALE)
        logging.info(f"Saved image at {new_img_path}")
        save_image(config, "png_cases", file, im, case)

def read_image(config, folder, filename, case):
    """
    Function that reads an image from directory
    """
    dir_path = os.path.join(*config["data_path"], folder, case)
    img_path = os.path.join(dir_path, filename + ".png")
    if os.path.exists(img_path) == False:
        logging.error(f"Image {img_path} does not exsist")
        exit()
    else:
        match folder:
            case _:
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img

def process_image(config, image_name):
    """
    Function that processes an image
    """


def save_image(config, folder, filename, img, case):
    """
    Function that saves images
    """
    dir_path = os.path.join(*config["data_path"], folder, case)
    if os.path.exists(dir_path) is False:
        os.makedirs(dir_path)
        logging.info(f"Creating dir {folder} for case {case}")
    img_path = os.path.join(*config["data_path"], folder, case, filename + ".png")
    match folder:
        case "png_cases":
            cv2.imwrite(img_path, img)
        case "binary":
            cv2.imwrite(img_path, img)
        case _:
            # mlt.imsave(img_path, img)
            cv2.imwrite(img_path, img)

def binarize_image(config, case):
    """
    Function that creates binarize images
    """
    images = get_image_files(config, case, "background_removed_cases")

    if images == []:
        logging.warning(f"No images found in folder background_removed_cases for case {case}")

    for image in images:
        new_img_path = os.path.join(*config["data_path"], "binary_cases", case, image + ".png")
        if os.path.exists(new_img_path):
            continue
        tmp_img = read_image(config, "background_removed_cases", image, case)
        # th, im_gray_th_otsu = cv2.threshold(tmp_img, 128, 255, cv2.THRESH_BINARY |cv2.THRESH_OTSU)
        th, im_gray_th_otsu = cv2.threshold(tmp_img, 80, 255, cv2.THRESH_OTSU)
        logging.info(f"Theshold for image {image}: {th}")
        pixels, px_count = np.unique(im_gray_th_otsu, return_counts=True)
        white_ratio = px_count[1]/px_count[0]
        logging.info(f"White ratio for image {image}: {white_ratio}")
        if white_ratio < 0.05:
            logging.warning(f"Binarization failed for image {image}")
            continue
        im_gray_th_otsu = cv2.bitwise_not(im_gray_th_otsu)
        save_image(config, "binary_cases", image, im_gray_th_otsu, case)

def substract_background(config, case):
    """
    Function that supstracts background from image
    """
    # logging.info(f"Substracting background from images for case {case}")
    images = get_image_files(config, case, "png_cases")
    imgs = config["images"]
    config["images"] = [0]
    background_img = get_image_files(config, case, "png_cases")
    config["images"] = imgs
    background = read_image(config, "png_cases", background_img[0], case)
    for img in images[1:]:
        new_img_path = os.path.join(*config["data_path"], "background_removed_cases", case, img + ".png")
        if os.path.exists(new_img_path):
            continue
        tmp_img = read_image(config, "png_cases", img, case)
        img_diff = cv2.absdiff(tmp_img, background)
        logging.info(f"Removed backgound for image {img}")
        save_image(config, "background_removed_cases", img, img_diff, case)

def make_histos(config, case, folder):
    """
    Function that creates histogram from image values
    """
    images = get_image_files(config, case, folder)

    for name in images:
        base_path=os.path.join(*config["data_path"], "histogramms", folder, case)
        if os.path.exists(base_path) == False:
            os.makedirs(base_path)
        hist_path = os.path.join(base_path, name + "_hist.png")
        if os.path.exists(hist_path):
            logging.info(f"Already created histogramm for image {name} in folder {folder}")
            continue
        img = read_image(config, folder, name, case)
        nums, counts = np.unique(img, return_counts=True)
        plt.hist(nums, nums, weights=counts)   
        plt.savefig(hist_path)
        logging.info(f"Created histogramm for image {name} in folder {folder}")
        plt.close()

def create_animation(config, case, folder):
    """
    Function that creates animation video for images
    """

    images = get_image_files(config, case, "png_cases")
    if images == []:
        logging.warning(f"No images found to create animation for case {case}.")
        return
    animation_folder_path = os.path.join(*config["data_path"], "animations", case)
    dat_path = os.path.join(*config["data_path"], folder, case)
    frame = cv2.imread(os.path.join(dat_path, images[0]+ ".png"))

    height, width, layers = frame.shape
    fps = 1000/500
    video_path = os.path.join(animation_folder_path, folder + ".avi")
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))
    for image in images:
        img = cv2.imread(os.path.join(dat_path, image + ".png"))
        out.write(img)

    cv2.destroyAllWindows()
    out.release()
    logging.info(f"Created {folder} video for case {case}")

def reset_cases(config, case):
    """
    Function to delete all processed images
    """
    folders = ["background_removed_cases", "binary_cases", "histogramms"]
    imgs = config["images"]
    for fold in folders:
        fld_path = os.path.join(*config["data_path"], fold)
        if os.path.exists(fld_path) == False:
            logging.warning(f"Folder {fld_path} does not exsist")
        else:
            shutil.rmtree(fld_path)
            logging.info(f"Removed folder {fold}")

if __name__ == "__main__":
    calc_case_ratio()