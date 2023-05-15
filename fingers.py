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

        for img in images:
            status = get_fingers(config, cas, img, background_img)

def get_fingers(config, case, img_name, background_img):
    """
    Method that gets the finger information for one image
    """

    # get background image array
    background_array = sitk.GetArrayFromImage(background_img)
    base_img = get_base_image(config, case, img_name)
    base_array = sitk.GetArrayFromImage(base_img)

    # substract bckground from image
    new_image = base_array - background_array

    # plt background substracted image
    fig, axs = plt.subplots()
    axs.set_title(f"Background substracted {img_name.split('_')[0]}")
    axs.imshow(new_image, cmap="Greys")
    if config["debug"] is False:
        plt.close(fig)
    else:
        plt.show()

    ots_img = sitk.GetImageFromArray(new_image)

    # # Load an image
    # image = ots_img

    # # Apply Otsu thresholding
    # otsu_filter = sitk.OtsuThresholdImageFilter()
    # otsu_threshold = otsu_filter.GetThreshold()
    # binary_image = sitk.BinaryThreshold(image, 0, otsu_threshold, 0, 1)
    
    # Define the lower and upper threshold values
    lower_threshold = 70
    upper_threshold = 220

    # connected region
    seed = [(1400,1200)]

    insta_image = sitk.ConnectedThreshold(
        image1=ots_img,
        seedList=seed,
        lower=lower_threshold,
        upper=upper_threshold,
        replaceValue=1
    )
    px_val = ots_img.GetPixel(seed[0])
    logging.info(f"px val = {px_val}")

    insta_proc_img = sitk.LabelOverlay(base_img, insta_image)

    

    # Create a binary threshold filter
    threshold_filter = sitk.ThresholdImageFilter()


    # Set the lower and upper thresholds
    threshold_filter.SetLower(lower_threshold)
    threshold_filter.SetUpper(upper_threshold)
    # threshold_filter.SetInsideValue(1)  # Value for pixels within the threshold range
    threshold_filter.SetOutsideValue(0)  # Value for pixels outside the threshold range

    # Apply the threshold filter to the input image
    output_image = threshold_filter.Execute(ots_img)

    output_image = sitk.BinaryThreshold(
        image1=ots_img,
        lowerThreshold=lower_threshold,
        upperThreshold=upper_threshold,
        insideValue=1,
        outsideValue=0
    )

    tmp_image = sitk.BinaryDilate(
        image1=output_image,
        backgroundValue=0.0,
        foregroundValue=1.0,
        boundaryToForeground=True,
        kernelRadius=(2,2)
    )

    tmp_image = sitk.BinaryErode(
        image1=tmp_image,
        backgroundValue=0.0,
        foregroundValue=1.0,
        boundaryToForeground=True,
        kernelRadius=(2,2)
    )

    fig, axs = plt.subplots()
    axs.set_title(f"Int method begin {img_name.split('_')[0]}")
    axs.imshow(sitk.GetArrayViewFromImage(output_image), cmap="Greys")
    if config["debug"] is False:
        plt.close(fig)
    else:
        plt.show()

    fig, axs = plt.subplots()
    axs.set_title(f"Int method end {img_name.split('_')[0]}")
    axs.imshow(sitk.GetArrayViewFromImage(tmp_image), cmap="Greys")
    if config["debug"] is False:
        plt.close(fig)
    else:
        plt.show()

    tmp_image = sitk.BinaryDilate(
        image1=output_image,
        backgroundValue=0.0,
        foregroundValue=1.0,
        boundaryToForeground=True,
        kernelRadius=(20,20)
    )

    tmp_image = sitk.BinaryErode(
        image1=tmp_image,
        backgroundValue=0.0,
        foregroundValue=1.0,
        boundaryToForeground=True,
        kernelRadius=(20,20)
    )

    fig, axs = plt.subplots()
    axs.set_title(f"Int method fingers closed {img_name.split('_')[0]}")
    axs.imshow(sitk.GetArrayViewFromImage(tmp_image), cmap="Greys")
    if config["debug"] is False:
        plt.close(fig)
    else:
        plt.show()

    fig, axs = plt.subplots()
    axs.set_title(f"Conected method {img_name.split('_')[0]}")
    axs.plot(1400,1200, marker="*", markersize=8, color="red")
    axs.imshow(sitk.GetArrayViewFromImage(insta_proc_img), cmap="Greys")
    if config["debug"] is False:
        plt.close(fig)
    else:
        plt.show()

if __name__ == "__main__":
    config = get_config()
    proc_cases(config)