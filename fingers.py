import os
import sys
import json
import glob
import logging

from multiprocessing import Pool
from functools import partial
from methods import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import SimpleITK as sitk

# setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s")


def proc_cases(config):

    for cas in list(config["cases"]):

        create_intermediate_finger_folders(config, cas)

        logging.info("-------------------")
        logging.info(f"Starting processing case {cas}")
        logging.info("-------------------")

        images = get_image_files(config, cas, "png_cases")
        if images == []:
            images = get_image_files(config, cas, "raw_cases")
        if images == []:
            logging.warning(f"No images found for case {cas}")
            continue
        images.sort()

        tmp_conf = config.copy()
        tmp_conf["images"] = []
        background_images = get_image_files(tmp_conf, cas, "png_cases")
        if background_images == []:
            background_images = get_image_files(tmp_conf, cas, "raw_cases")
        if background_images == []:
            logging.warning(f"No images found for case {cas}")
            continue

        background_img = get_base_image(config, cas, background_images[0])
        res = {
            "image" : [],
            "ratio" : []
        }
        
        if config["debug"]:
            parallel = False
        else:
            parallel = True

        if parallel:
            cpus = os.cpu_count()
            p = Pool(cpus)
            for rat in p.map(partial(get_fingers, config=config, case=cas, background_img=background_img), images):
                logging.debug(f"Multi res : {rat}")
                res["image"].append(int(rat[0].split('_')[0]))
                res["ratio"].append(rat[1])
        else:
            for img in images:
                name, ratio = get_fingers(img, config, cas, background_img=background_img)
                res["image"].append(int(name.split('_')[0]))
                res["ratio"].append(ratio)
            
        fig, axs = plt.subplots()
        axs.set_title("Finger ratio")
        axs.plot(res["image"], res["ratio"])
        fig_path = os.path.join(config["results_path"], "finger_data", cas, "plots", "ratio.png")
        plt.savefig(fig_path)
        logging.info(f"Saved ratio fig at {fig_path}")
        plt.show()

        if len(config["images"]) == 0:
            csv_path = os.path.join(config["results_path"], "finger_data", cas, "ratio", "ratio.csv")
            df = pd.DataFrame(res)
            df.to_csv(csv_path, index=False)
            logging.info(f"Saved ratio data at {csv_path}")



def save_intermediate(config, case, img_name, array, folder):
    """
    Function that saves intermediate to folder
    """
    im_path = os.path.join(config["data_path"], folder, case, img_name + ".png")
    plt.imsave(im_path, array, cmap="Greys", dpi=1200)


def substract_background(config, case, img_name, background_img) -> sitk.Image:
    """
    Function that substracts background (first image of series) from current image 
    """
    img_path = os.path.join(config["data_path"], "background_substracted", case, img_name + ".png")
    if os.path.exists(img_path) and config["new_files"] is False:
        logging.info(f"Already did background substraction for image {img_name}")
        reader = sitk.ImageFileReader()
        reader.SetImageIO("PNGImageIO")
        reader.SetFileName(img_path)
        image = reader.Execute()
        return image

    # get background image array
    background_array = sitk.GetArrayFromImage(background_img)
    base_img = get_base_image(config, case, img_name)
    base_array = sitk.GetArrayFromImage(base_img)

    # substract bckground from image
    new_image = base_array - background_array

    new_image = delete_border(sitk.GetImageFromArray(new_image))

    # plt background substracted image
    fig, axs = plt.subplots()
    axs.set_title(f"Background substracted {img_name.split('_')[0]}")
    axs.imshow(sitk.GetArrayFromImage(new_image))
    if config["debug"] is False:
        plt.close(fig)
    else:
        plt.show()

    save_intermediate(config, case, img_name, sitk.GetArrayFromImage(new_image), "background_substracted")

    return new_image

def delete_border(img) -> sitk.Image:
    """
    Function that sets the outer border values of image to 0
    """
    img_array = sitk.GetArrayFromImage(img)
    offset = 100
    img_array[:offset, :] = 0
    img_array[img_array.shape[0]- offset: img_array.shape[0], :] = 0
    img_array[:,:offset] = 0
    img_array[:, img_array.shape[1]- offset: img_array.shape[1]] = 0

    final_img = sitk.GetImageFromArray(img_array)
    
    return final_img

def create_intermediate_finger_folders(config, cas):
    """
    Function that creates all intermediate folders
    """
    folders = ["background_substracted", "finger_image", "int_window", "cleaned_artifacts", "closed_fingers"]

    for fld in folders:

        folder_path = os.path.join(config["data_path"], fld, cas)
        if os.path.exists(folder_path) is False:
            os.makedirs(folder_path)

    # create folders for results
    final_folders = ["finger_data"]
    for fld in final_folders:
        folder_path = os.path.join(config["results_path"], fld, cas)
        if os.path.exists(folder_path) is False:
            os.makedirs(folder_path)
        folder_path = os.path.join(config["results_path"], fld, cas, "ratio")
        if os.path.exists(folder_path) is False:
            os.makedirs(folder_path)
        folder_path = os.path.join(config["results_path"], fld, cas, "plots")
        if os.path.exists(folder_path) is False:
            os.makedirs(folder_path)

def int_window(config, case, img_name, base_image):
    """
    Function that applys intensity window to image
    """

    img_path = os.path.join(config["data_path"], "int_window", case, img_name + ".png")
    if os.path.exists(img_path) and config["new_files"] is False:
        logging.info(f"Already did intensity window for image {img_name}")
        reader = sitk.ImageFileReader()
        reader.SetImageIO("PNGImageIO")
        reader.SetFileName(img_path)
        image = reader.Execute()
        return image

    lower_threshold = 70
    upper_threshold = 250

    # Create a binary threshold filter
    threshold_filter = sitk.ThresholdImageFilter()

    # Set the lower and upper thresholds
    threshold_filter.SetLower(lower_threshold)
    threshold_filter.SetUpper(upper_threshold)
    # threshold_filter.SetInsideValue(1)  # Value for pixels within the threshold range
    threshold_filter.SetOutsideValue(0)  # Value for pixels outside the threshold range

    # Apply the threshold filter to the input image
    output_image = threshold_filter.Execute(base_image)

    output_image = sitk.BinaryThreshold(
        image1=output_image,
        lowerThreshold=lower_threshold,
        upperThreshold=upper_threshold,
        insideValue=1,
        outsideValue=0
    )

    fig, axs = plt.subplots()
    axs.set_title(f"Intensity window {img_name.split('_')[0]}")
    axs.imshow(sitk.GetArrayViewFromImage(output_image), cmap="Greys")
    if config["debug"] is False:
        plt.close(fig)
    else:
        plt.show()

    save_intermediate(config, case, img_name, sitk.GetArrayFromImage(output_image), "int_window")

    return output_image

def clean_artifacts(config, case, img_name, base_image) -> sitk.Image:
    """
    Function that gets rid of small artifacts within the image by Erosion and Dilation
    """

    img_path = os.path.join(config["data_path"], "cleaned_artifacts", case, img_name + ".png")
    if os.path.exists(img_path) and config["new_files"] is False:
        logging.info(f"Already clean artifacts for image {img_name}")
        reader = sitk.ImageFileReader()
        reader.SetImageIO("PNGImageIO")
        reader.SetFileName(img_path)
        image = reader.Execute()
        return image

    px_len = 5

    # get rid of small artifacts by dilation and erosion
    tmp_image = sitk.BinaryErode(
        image1=base_image,
        backgroundValue=0.0,
        foregroundValue=1.0,
        boundaryToForeground=True,
        kernelRadius=(px_len,px_len)
    )

    tmp_image = sitk.BinaryDilate(
        image1=tmp_image,
        backgroundValue=0.0,
        foregroundValue=1.0,
        boundaryToForeground=True,
        kernelRadius=(px_len,px_len)
    )

    # get rid of border artifacts created from cleaning
    tmp_image = delete_border(tmp_image)

    fig, axs = plt.subplots()
    axs.set_title(f"Cleaned artifacts {img_name.split('_')[0]}")
    axs.imshow(sitk.GetArrayViewFromImage(tmp_image), cmap="Greys")
    if config["debug"] is False:
        plt.close(fig)
    else:
        plt.show()

    save_intermediate(config, case, img_name, sitk.GetArrayFromImage(tmp_image), "cleaned_artifacts")
    
    return tmp_image

def close_finger(config, case, img_name, base_img) -> sitk.Image:
    """
    Function that closes all the fingers within an image
    """

    img_path = os.path.join(config["data_path"], "closed_fingers", case, img_name + ".png")
    if os.path.exists(img_path) and config["new_files"] is False:
        logging.info(f"Already closed fingers for image {img_name}")
        reader = sitk.ImageFileReader()
        reader.SetImageIO("PNGImageIO")
        reader.SetFileName(img_path)
        image = reader.Execute()
        return image
    

    px_len = 20

    tmp_image = sitk.BinaryDilate(
        image1=base_img,
        backgroundValue=0.0,
        foregroundValue=1.0,
        boundaryToForeground=True,
        kernelRadius=(px_len,px_len)
    )

    tmp_image = sitk.BinaryErode(
        image1=tmp_image,
        backgroundValue=0.0,
        foregroundValue=1.0,
        boundaryToForeground=True,
        kernelRadius=(px_len,px_len)
    )

    fig, axs = plt.subplots()
    axs.set_title(f"Finger closed {img_name.split('_')[0]}")
    axs.imshow(sitk.GetArrayViewFromImage(tmp_image), cmap="Greys")
    if config["debug"] is False:
        plt.close(fig)
    else:
        plt.show()

    save_intermediate(config, case, img_name, sitk.GetArrayFromImage(tmp_image), "closed_fingers")

    return tmp_image

def diff_image(config, case, img_name, cleaned_image, closed_fingers, base_image):
    """
    Function that plots the finger image
    """

    calc_img = closed_fingers - cleaned_image

    # diff_img = sitk.LabelOverlay(base_image, calc_img)
    diff_img = calc_img

    fig, axs = plt.subplots()
    axs.set_title(f"Finger diff {img_name.split('_')[0]}")
    axs.imshow(sitk.GetArrayViewFromImage(diff_img), cmap="Greys")
    if config["debug"] is False:
        plt.close(fig)
    else:
        plt.show()

    save_intermediate(config, case, img_name, sitk.GetArrayFromImage(diff_img), "finger_image")

    return diff_img


def get_fingers(img_name, config, case, background_img) -> float:
    """
    Method that gets the finger information for one image
    """
    logging.info(f"Substracting background for image {img_name}")
    base_image = substract_background(config, case, img_name, background_img)

    logging.info(f"Applying intensity window for image {img_name}")
    windowed_image = int_window(config, case, img_name, base_image)
    
    logging.info(f"Cleaning artifacts for image {img_name}")
    cleaned_image = clean_artifacts(config,case, img_name, windowed_image)

    logging.info(f"Closing fingers for image {img_name}")
    closed_fingers = close_finger(config,case, img_name, cleaned_image)

    logging.info(f"Calculating finger ratio for {img_name}")
    fing_px_val, fing_px_n = np.unique(sitk.GetArrayFromImage(cleaned_image), return_counts=True)
    cls_px_val, cls_px_n = np.unique(sitk.GetArrayFromImage(closed_fingers), return_counts=True)

    logging.info(f"Calculating diff image for {img_name}")
    diff_img = diff_image(config, case, img_name, cleaned_image, closed_fingers, base_image)

    if len(fing_px_val) > 1 and len(cls_px_val) > 1:
        ratio = round(fing_px_n[1]/cls_px_n[1], 3)
        logging.info(f"Finger ratio for {img_name}: {ratio}")
        return img_name, ratio
    else:
        logging.warning(f"No ratio calculation possible for image {img_name}")
        return img_name, 0.0
if __name__ == "__main__":
    config = get_config()
    proc_cases(config)