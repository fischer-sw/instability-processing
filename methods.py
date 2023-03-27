import os
import sys
import json
import glob
import logging
import shutil

from multiprocessing import Pool
from functools import partial

import numpy as np
import pandas as pd
import PIL
import matplotlib.pyplot as plt
import matplotlib.image as mlt
import cv2
import SimpleITK as sitk

# setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s")


def get_image_files(config, case, folder):
    """
    Function that reads image names from folder
    """
    dir_path = os.path.join(config["data_path"])
    if os.path.exists(dir_path) == False:
        logging.error(f"Data directory {dir_path} doesn't exsist. Please check config for valid path.")
        exit()

    dat_path = os.path.join(config["data_path"], folder, case)
    if os.path.exists(dat_path) == False:
        logging.warning(f"No data found for case {case} at {dat_path}")
        return []
    images = glob.glob(os.path.join(dat_path, "*.tiff"))
    if images == []:
        images = glob.glob(os.path.join(dat_path, "*.png"))
    if images == []:
        logging.warning(f"No images found in folder {folder} for case {case}")
        tmp_images = []
    else:
        images = [os.path.basename(x) for x in images]
        tmp_images = [x.split(".")[0] for x in images]

    logging.info('found {} images'.format(len(tmp_images)))
    if config["images"] != []:
        img_subset = []
        # append first image (background)
        # if not 0 in config["images"]:
        #     config["images"].append(0)
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

        parallel = True

        if parallel:
            cpus = os.cpu_count()
            p = Pool(cpus)
            p.map(partial(process_image, config=config, cas=cas), images)
        else:
            for img in images:
                status = process_image(img, config, cas)
            
    # create_animation(config, cas, "png_cases")
    # create_animation(config, cas, "binary_cases")

def get_all_cases(config):
    """
    Function that puts all cases in config
    """
    path = os.path.join(config["data_path"], "raw_cases")
    dirs = os.listdir(path)
    config["all_cases"] = dirs
    cfg_path = os.path.join(sys.path[0], "config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)    


def convert2png(config, case, file) -> bool:
    """
    Function that converst .tiff images to .png images and stores them in png_cases folder for a case
    """
    
    new_dir_path = os.path.join(config["data_path"], "png_cases", case)
    new_img_path = os.path.join(new_dir_path, file + ".png")
    img_path = os.path.join(config["data_path"], "raw_cases", case, file)
    if os.path.exists(new_dir_path) is False:
        os.makedirs(new_dir_path)
    if os.path.exists(new_img_path):
        return True
    # im = Image.open(img_path + ".tiff")
    im = cv2.imread(img_path + ".tiff", cv2.IMREAD_GRAYSCALE)
    logging.info(f"Saved image at {new_img_path}")
    save_image(config, "png_cases", file, im, case)
    return True

def read_image(config, folder, filename, case):
    """
    Function that reads an image from directory
    """
    dir_path = os.path.join(config["data_path"], folder, case)
    img_path = os.path.join(dir_path, filename + ".png")
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        logging.error(f"Image {img_path} does not exsist")
        exit()

    return img

def process_image(img_name, config, cas) -> bool:
    """
    Function that processes an image
    """

    # segment_camera(config, cas, img_name)
    status = False
    status = convert2png(config, cas, img_name)
    # if status:
    #     hist_stat = make_histo(config, cas, "png_cases", img_name)
    # else:
    #     logging.error(f"Converting image {img_name} to png failed")
    #     return status
    # status = substract_background(config, cas, img_name)
    # if status:
    #     hist_stat = make_histo(config, cas, "background_removed_cases", img_name)
    # else:
    #     logging.error(f"Background substraction for image {img_name} failed")
    #     return status

    # if int(img_name.split("_")[0]) == 0:
    #     return False

    # status = binarize_image(config, cas, img_name)
    # if status:
    #     hist_stat = make_histo(config, cas, "binary_cases", img_name)
    # else:
    #     logging.error(f"Binarization for image {img_name} failed")
    #     return status

    # status = detect_circles(config, cas, "binary_cases", img_name)

def detect_circles(config, case, folder, image_name):
    """
    Function to detect circle within a binary image
    """
    img = read_image(config, folder, image_name, case)

    h, w = img.shape[:2]
    # Detect circles in the image
    circles = cv2.HoughCircles(img,
        cv2.HOUGH_GRADIENT,
        1.2,
        20,
        param1=50,
        param2=30,
        minRadius=100,
        maxRadius=w
    )
    if circles != None:
        logging.info(f"Circles = {circles}")

def save_image(config, folder, filename, img, case):
    """
    Function that saves images
    """
    dir_path = os.path.join(config["data_path"], folder, case)
    if os.path.exists(dir_path) is False:
        os.makedirs(dir_path)
        logging.info(f"Creating dir {folder} for case {case}")
    img_path = os.path.join(config["data_path"], folder, case, filename + ".png")
    cv2.imwrite(img_path, img)

def crop(file_in, bbox, file_out):
    from PIL import Image
    image = Image.open(file_in)
    crop = image.crop(box=bbox)
    crop.save(file_out)

def edges(file_in, file_out):
    from PIL import Image, ImageFilter
    image = Image.open(file_in)
    result = image.filter(ImageFilter.FIND_EDGES)
    result.save(file_out)

def enhance(file_in, factor, file_out):
    from PIL import Image, ImageEnhance
    image = Image.open(file_in)
    enhancer = ImageEnhance.Sharpness(image)
    result = enhancer.enhance(factor)
    result.save(file_out)

def segment_camera(config, case, img_name):
    """
    Function that segements camera from the image and moves it to background
    """

    folder = "segmented_camera"
    base_path=os.path.join(config["data_path"], folder, case)
    if os.path.exists(base_path) == False:
        os.makedirs(base_path)
    new_img_path = os.path.join(config["data_path"], folder, case, img_name + ".png")
    if os.path.exists(new_img_path):
        return True

    raw_img_path = os.path.join(config["data_path"], "png_cases", case, img_name + ".png")
    if os.path.exists(raw_img_path):
        reader = sitk.ImageFileReader()
        reader.SetImageIO("PNGImageIO")
        reader.SetFileName(raw_img_path)
        raw_img = reader.Execute()
    else:
        raw_img_path = os.path.join(config["data_path"], "raw_cases", case, img_name + ".tiff")
        reader = sitk.ImageFileReader()
        reader.SetImageIO("TIFFImageIO")
        reader.SetFileName(raw_img_path)
        raw_img = reader.Execute()

    # crop(raw_img_path, (1000, 630, 1800, 1430), 'crop.png')
    crop(raw_img_path, (1000, 630, 1800, 1000), 'crop.png')

    #factor = 10.0
    #enhance('crop.png', factor, 'enhance_{}.png'.format(factor))
    #reader.SetFileName('enhance_{}.png'.format(factor))
    # edges('crop.png', 'edges.png')
    reader.SetFileName('crop.png')
    crop_img = reader.Execute()

    crop_array = sitk.GetArrayFromImage(crop_img)
    logging.info('crop shape {}'.format(crop_array.shape))

    cam_image = sitk.ConnectedThreshold(
        image1=crop_img,
        seedList=[(370,460)],
        lower=0,
        upper=80,
        replaceValue=1
    )
    uni_vals = np.unique(sitk.GetArrayFromImage(cam_image))
    logging.info(f"Unique values {uni_vals}")

    cam_proc_img = sitk.LabelOverlay(crop_img, cam_image)
    
    #gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
    #gaussian.SetSigma(float(40.0))
    #blur_image = gaussian.Execute(raw_img)

    #caster = sitk.CastImageFilter()
    #caster.SetOutputPixelType(raw_img.GetPixelID())
    #blur_image = caster.Execute(blur_image)


    # raw_img = raw_img - blur_image
    # raw_img = blur_image

    insta_seed = [(570,460)]
    px_val = crop_img.GetPixel(insta_seed[0])
    logging.info(f"Insta seed value: {px_val}")

    # uni_vals = np.unique(sitk.GetArrayFromImage(raw_img))
    # logging.info(f"Unique values blured img {uni_vals}")

    # upper halt: [ 100, 224 ]
    # lower right quadrant: [ 100, 230 ]
    # lower left quad [ 100, 235 ]
    insta_image = sitk.ConnectedThreshold(
        image1=crop_img,
        seedList=insta_seed,
        lower=100,
        upper=224,
        replaceValue=2
    )
    uni_vals = np.unique(sitk.GetArrayFromImage(insta_image))
    logging.info(f"Unique values {uni_vals}")

    insta_proc_img = sitk.LabelOverlay(crop_img, insta_image)

    # sitk_show(raw_img)

    writer = sitk.ImageFileWriter()
    writer.SetFileName("cam_segmented.png")
    writer.Execute(cam_proc_img)

    #writer = sitk.ImageFileWriter()
    #writer.SetFileName("crop.png")
    #writer.Execute(crop_img)

    writer = sitk.ImageFileWriter()
    writer.SetFileName("insta_segmented.png")
    writer.Execute(insta_proc_img)

    writer = sitk.ImageFileWriter()
    writer.SetFileName("raw.png")
    writer.Execute(raw_img)

def binarize_image(config, case, img_name) -> bool:
    """
    Function that creates binarize images
    """
    new_img_path = os.path.join(config["data_path"], "binary_cases", case, img_name + ".png")
    if os.path.exists(new_img_path):
        return True
    tmp_img = read_image(config, "background_removed_cases", img_name, case)
    # th, im_gray_th_otsu = cv2.threshold(tmp_img, 128, 255, cv2.THRESH_BINARY |cv2.THRESH_OTSU)
    th, im_gray_th_otsu = cv2.threshold(tmp_img, 80, 255, cv2.THRESH_OTSU)
    logging.info(f"Theshold for image {img_name}: {th}")
    pixels, px_count = np.unique(im_gray_th_otsu, return_counts=True)
    white_ratio = px_count[1]/px_count[0]
    logging.info(f"White ratio for image {img_name}: {white_ratio}")
    if white_ratio < 0.05:
        logging.warning(f"Binarization failed for img_name {img_name}")
        return False
    im_gray_th_otsu = cv2.bitwise_not(im_gray_th_otsu)
    save_image(config, "binary_cases", img_name, im_gray_th_otsu, case)
    return True

def substract_background(config, case, img_name) -> bool:
    """
    Function that supstracts background from image
    """
    # logging.info(f"Substracting background from images for case {case}")
    new_img_path = os.path.join(config["data_path"], "background_removed_cases", case, img_name + ".png")
    if os.path.exists(new_img_path):
        return True
    imgs = config["images"]
    config["images"] = [0]
    background_img = get_image_files(config, case, "png_cases")
    config["images"] = imgs
    background = read_image(config, "png_cases", background_img[0], case)
    tmp_img = read_image(config, "png_cases", img_name, case)
    img_diff = cv2.absdiff(tmp_img, background) * 10
    logging.info(f"Removed backgound for image {img_name}")
    save_image(config, "background_removed_cases", img_name, img_diff, case)
    return True

def make_histo(config, case, folder, name) -> bool:
    """
    Function that creates histogram from image values
    """
    
    base_path=os.path.join(config["data_path"], "histogramms", folder, case)
    if os.path.exists(base_path) == False:
        os.makedirs(base_path)
    hist_path = os.path.join(base_path, name + "_hist.png")
    if os.path.exists(hist_path):
        logging.info(f"Already created histogramm for image {name} in folder {folder}")
        return True
    img = read_image(config, folder, name, case)
    nums, counts = np.unique(img, return_counts=True)
    plt.hist(nums, nums, weights=counts)   
    plt.savefig(hist_path)
    logging.info(f"Created histogramm for image {name} in folder {folder}")
    plt.close()
    return True

def create_animation(config, case, folder):
    """
    Function that creates animation video for images
    """

    images = get_image_files(config, case, "png_cases")
    if images == []:
        logging.warning(f"No images found to create animation for case {case}.")
        return
    animation_folder_path = os.path.join(config["data_path"], "animations", case)
    dat_path = os.path.join(config["data_path"], folder, case)
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
        fld_path = os.path.join(config["data_path"], fold)
        if os.path.exists(fld_path) == False:
            logging.warning(f"Folder {fld_path} does not exsist")
        else:
            shutil.rmtree(fld_path)
            logging.info(f"Removed folder {fold}")

if __name__ == "__main__":
    calc_case_ratio()
    # get_all_cases(get_config())