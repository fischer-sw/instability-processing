import os
import sys
import json
import glob
import logging
import shutil

from multiprocessing import Pool
from functools import partial
from PIL import Image

import numpy as np
import pandas as pd
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

        # images = get_image_files(config, cas, "raw_cases")
        images = get_image_files(config, cas, "png_cases")
        images.sort()

        if config["debug"]:
            parallel = False
        else:    
            parallel = True

        if parallel:
            cpus = os.cpu_count()
            p = Pool(cpus)
            p.map(partial(process_image, config=config, cas=cas), images)
        else:
            for img in images:
                status = process_image(img, config, cas)
            
        create_animation(config, cas, "final_results")
        create_animation(config, cas, "png_cases")

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
    final_dir_path = os.path.join(config["results_path"], "final_data", cas)
    if os.path.exists(final_dir_path) == False:
        os.makedirs(final_dir_path)
    res_csv_folder = os.path.join(final_dir_path, "instabilities")
    if os.path.exists(res_csv_folder) == False:
        os.makedirs(res_csv_folder)
    res_csv_path = os.path.join(res_csv_folder ,img_name + ".csv")
    if os.path.exists(res_csv_path) and config["new_files"] == False:
        logging.info(f"Already processed image {img_name} for case {cas}")
        return True

    status = False
    status = convert2png(config, cas, img_name)
    if status:
        hist_stat = make_histo(config, cas, "png_cases", img_name)
    else:
        logging.error(f"Converting image {img_name} to png failed")
        return status

    base_img = get_base_image(config, cas, img_name)

    base_array = sitk.GetArrayFromImage(base_img)
    base_array[:,500] = [0] * base_array.shape[0]
    base_img = sitk.GetImageFromArray(base_array)

    cam_img = segment_camera(config, cas, base_img, img_name)

    # get cam array
    cam_array = sitk.GetArrayFromImage(cam_img)
    # replace all found cam pixel with 0 in base image
    base_array[cam_array == 1] = 0
    base_img = sitk.GetImageFromArray(base_array)
    insta_img, status = segment_instability(config, cas, base_img, img_name)
    if status == False:
        logging.error(f"No instability found in {img_name}")
        return status

    # hull = convex_hull(config, cas, insta_img, img_name)
    new_insta_img = refine_instability(config, cas, insta_img, img_name)
    final_insta_img = close_instability(config, cas, new_insta_img, img_name)
    
    contours = get_contour(config, cas, final_insta_img, img_name)
    df = pd.DataFrame(contours)  
    con_csv_folder = os.path.join(final_dir_path, "contours")
    if os.path.exists(con_csv_folder) == False:
        os.makedirs(con_csv_folder)
    con_csv_path = os.path.join(con_csv_folder, img_name + ".csv")
    df.to_csv(con_csv_path)

    res_array = sitk.GetArrayFromImage(final_insta_img)
    final_img_path = os.path.join(config["data_path"], "final_images", cas)
    if os.path.exists(final_img_path) == False:
        os.makedirs(final_img_path)
    img_path = os.path.join(final_img_path, img_name + ".png")
    plt.imsave(img_path, res_array, cmap="Greys", dpi=1200)
    df = pd.DataFrame(res_array)
    df.to_csv(res_csv_path)

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

def get_base_image(config, case, file_name):
    """
    Function that gets base image to process
    """
    raw_img_path = os.path.join(config["data_path"], "png_cases", case, file_name + ".png")
    if os.path.exists(raw_img_path):
        reader = sitk.ImageFileReader()
        reader.SetImageIO("PNGImageIO")
        reader.SetFileName(raw_img_path)
        raw_image = reader.Execute()
    else:
        raw_img_path = os.path.join(config["data_path"], "raw_cases", case, file_name + ".tiff")
        reader = sitk.ImageFileReader()
        reader.SetImageIO("TIFFImageIO")
        reader.SetFileName(raw_img_path)
        raw_image = reader.Execute()

    return raw_image

def get_contour(config, case, base_image, file_name):
    """
    Function that gets the contour of the instability
    """
    # inv_base_img = sitk.InvertIntensity(base_image, maximum=1)

    base_array = sitk.GetArrayFromImage(base_image)
    # base_array -= 1

    contours, hierarchy = cv2.findContours(image=base_array, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

    contour_image = cv2.drawContours(image=np.zeros(base_array.shape), contours=contours, contourIdx=-1, color=(1, 0, 0), thickness=7, lineType=cv2.LINE_AA)
    # cv2.imshow("Contour image", contour_image)
    fig, axs = plt.subplots()
    axs.set_title(f"Contour Image {file_name.split('_')[0]}")
    pos = axs.imshow(contour_image, cmap="Greys")
    fig.colorbar(pos, ax=axs)

    folder = "contour"
    dir_path = os.path.join(config["data_path"], folder, case)
    if os.path.exists(dir_path) is False:
        os.makedirs(dir_path)
        logging.info(f"Creating dir {folder} for case {case}")
    if config["save_intermediate"]:
        fig.savefig(os.path.join(dir_path, file_name + ".png"))
    if config["debug"] == False:
        plt.close(fig)
    return contour_image

def close_instability(config, case, base_image, file_name):
    """
    Function that trys to close camera hole. WORK IN PROGRESS!
    """
    image_array = sitk.GetArrayFromImage(base_image)
    lower = (0,0)
    upper = (0,0)
    rates = {
        "line" : [],
        "values" : []
    }
    contours, hierarchy = cv2.findContours(image=image_array, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    cont = contours[-1]

    M = cv2.moments(cont)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    logging.info(f"cX: {cX}, cY: {cY}")

    # upper half
    upper_x = image_array.shape[1]
    lower_x = upper_x
    upper_lines = list(range(0,cY))
    upper_lines.reverse()
    lower_lines = list(range(cY, image_array.shape[0]))

    for i in upper_lines:

        horiz_line_vals = image_array[i,:]
        px_vals, px_count = np.unique(horiz_line_vals, return_counts=True)
        if len(px_vals) < 2:
            break
        # logging.info(line_vals)
        black_pxs = px_count[1]
        rates["line"].append(i)
        rates["values"].append(px_count[1])
        tmp_x = np.where(horiz_line_vals == 1)[0].min()
        if tmp_x < upper_x:
            upper_y = i
            upper_x = tmp_x
            # logging.info(f"New Point ({upper_x}, {upper_y})")

    lines = False
    if lines:
        # vertical line
        line_thickness = 8
        verti = cY
        start = round(verti - line_thickness/2)
        markers = [start+ n*1 for n in range(line_thickness)]
        image_array[:,markers] = 1.0

        # horizontal line
        horiz = cX
        start = round(horiz - line_thickness/2)
        markers = [start+ n*1 for n in range(line_thickness)]
        image_array[markers,:] = 1.0

    upper = (int(upper_x), int(upper_y))


    for i in lower_lines:
        horiz_line_vals = image_array[i,:]
        px_vals, px_count = np.unique(horiz_line_vals, return_counts=True)
        if len(px_vals) < 2:
            break
        # logging.info(line_vals)
        black_pxs = px_count[1]
        rates["line"].append(i)
        rates["values"].append(px_count[1])
        tmp_x = int(np.where(horiz_line_vals == 1)[0].min())
        if tmp_x < lower_x:
            lower_y = i
            lower_x = tmp_x
            # logging.info(f"New Point ({lower_x}, {lower_y})")
    lower = (lower_x, lower_y)
        
    lines = False
    if lines:
        # vertical line
        line_thickness = 8
        verti = lower_x
        start = round(verti - line_thickness/2)
        markers = [start+ n*1 for n in range(line_thickness)]
        image_array[:,markers] = 1.0

        # horizontal line
        horiz = lower_lines[0]
        start = round(horiz - line_thickness/2)
        markers = [start+ n*1 for n in range(line_thickness)]
        image_array[markers,:] = 1.0  
    
    if upper != (0,0) and lower != (0,0):
        image_array = cv2.line(image_array, lower, upper, color=(1,0,0), thickness=5)
    

    fig, axs = plt.subplots()
    axs.set_title(f"Closed instability")
    pos = axs.imshow(image_array, cmap="Greys")
    fig.colorbar(pos, ax=axs)
    if config["debug"] == False:
        plt.close(fig)

    tmp_image = sitk.GetImageFromArray(image_array)

    # cam_seed = [(cX,cY)]
    cam_seed = [(1300,1300)]


    px_val = tmp_image.GetPixel(cam_seed[0])
    logging.info(f"Seed value: {px_val}")
    insta_image = sitk.ConnectedThreshold(
        image1=tmp_image,
        seedList=cam_seed,
        lower=0,
        upper=0.9,
        replaceValue=1
    )
    insta_array = sitk.GetArrayFromImage(insta_image)

    image_array = image_array + insta_array
    fig, axs = plt.subplots()
    axs.set_title(f"Filled instability")
    pos = axs.imshow(image_array, cmap="Greys")
    fig.colorbar(pos, ax=axs)
    if config["debug"] == False:
        plt.close(fig)
    
    folder = "closed_instability"
    dir_path = os.path.join(config["data_path"], folder, case)
    if os.path.exists(dir_path) is False:
        os.makedirs(dir_path)
        logging.info(f"Creating dir {folder} for case {case}")
    if config["save_intermediate"]:
        fig.savefig(os.path.join(dir_path, file_name + ".png"))
        # plt.imsave(os.path.join(dir_path, file_name + ".png"), image_array, cmap="Greys", dpi=1200)
    if config["debug"] == False:
        plt.close(fig)

    return sitk.GetImageFromArray(image_array)

def refine_instability(config, case, base_image, file_name):
    """
    Function that fills holes in instability segmentation
    """
    tmp_image = base_image

    tmp_image = sitk.BinaryErode(
        image1=tmp_image,
        backgroundValue=0.0,
        foregroundValue=1.0,
        boundaryToForeground=True,
        kernelRadius=(1,2)
    )

    tmp_image = sitk.BinaryDilate(
        image1=tmp_image,
        backgroundValue=0.0,
        foregroundValue=1.0,
        boundaryToForeground=True,
        kernelRadius=(1,2)
    )

    # get rid of small artifacts    
    tmp_image = sitk.BinaryErode(
            image1=tmp_image,
            backgroundValue=0.0,
            foregroundValue=1.0,
            kernelRadius=(7,7)
         )
    
    tmp_image = sitk.BinaryDilate(
            image1=tmp_image,
            backgroundValue=0.0,
            foregroundValue=1.0,
            kernelRadius=(7,7)
         )
    
    fig, axs = plt.subplots()
    axs.set_title(f"Instability after Dilate/Erode")
    pos = axs.imshow(sitk.GetArrayViewFromImage(tmp_image), cmap="Greys")
    fig.colorbar(pos, ax=axs)
    
    folder = "instability"
    dir_path = os.path.join(config["data_path"], folder, case)
    if os.path.exists(dir_path) is False:
        os.makedirs(dir_path)
        logging.info(f"Creating dir {folder} for case {case}")
    if config["save_intermediate"]:
        fig.savefig(os.path.join(dir_path, file_name + ".png"))
    if config["debug"] == False:
        plt.close(fig)
    return tmp_image

def convex_hull(config, case, base_image, file_name):

    tmp_image = base_image

    image = sitk.GetArrayFromImage(tmp_image)
    
    import skimage as ski

    chull = ski.morphology.convex_hull_image(image)
    fig, axs = plt.subplots()
    axs.set_title("Convex Hull")
    pos = axs.imshow(chull, cmap="Greys")
    fig.colorbar(pos, ax=axs)

    folder = "convex_hulls"
    dir_path = os.path.join(config["data_path"], folder, case)
    if os.path.exists(dir_path) is False:
        os.makedirs(dir_path)
        logging.info(f"Creating dir {folder} for case {case}")
    if config["save_intermediate"]:
        fig.savefig(os.path.join(dir_path, file_name + ".png"))
    if config["debug"] == False:
        plt.close(fig)

    return chull


def segment_camera(config, case, base_image, file_name):
    """
    Function that segements instability from an image
    """
    cam_seed = [(10,10)]

    px_val = base_image.GetPixel(cam_seed[0])
    logging.info(f"Cam seed value: {px_val}")
    cam_image = sitk.ConnectedThreshold(
        image1=base_image,
        seedList=cam_seed,
        lower=0,
        upper=55,
        replaceValue=1
    )
    uni_vals = np.unique(sitk.GetArrayFromImage(cam_image))
    logging.info(f"Unique values cam {uni_vals}")

    cam_proc_img = sitk.LabelOverlay(base_image, cam_image)
    fig, axs = plt.subplots()
    axs.set_title(f"Cam Image {file_name.split('_')[0]}")
    axs.imshow(sitk.GetArrayViewFromImage(cam_proc_img))

    folder = "segmented_camera"
    dir_path = os.path.join(config["data_path"], folder, case)
    if os.path.exists(dir_path) is False:
        os.makedirs(dir_path)
        logging.info(f"Creating dir {folder} for case {case}")
    if config["save_intermediate"]:
        fig.savefig(os.path.join(dir_path, file_name + ".png"))
    if config["debug"] == False:
        plt.close(fig)
    return cam_image

def segment_instability(config, case, base_image, file_name):
    """
    Function that segements camera from base image
    """
    status = True
    insta_seed = [(1450,1200)]
    px_val = base_image.GetPixel(insta_seed[0])
    logging.info(f"Insta seed value: {px_val}")

    # uni_vals = np.unique(sitk.GetArrayFromImage(base_image))
    # logging.info(f"Unique values blured img {uni_vals}")

    # upper halt: [ 100, 224 ]
    # lower right quadrant: [ 100, 230 ]
    # lower left quad [ 100, 235 ]

    lower_limit = 1
    if px_val > 230:
        return base_image, False
    else:
        upper_start = px_val + 10
    # initial segmentation
    insta_image = sitk.ConnectedThreshold(
            image1=base_image,
            seedList=insta_seed,
            lower=lower_limit,
            upper=upper_start,
            replaceValue=1
        )
    px_n, px_count = np.unique(sitk.GetArrayFromImage(insta_image), return_counts=True)

    if len(px_n) == 1:
        return base_image, False
    
    px_count_old = px_count[1]
    delta = px_count[1] / px_count_old
    i = 0
    step = 5
    delta_data = {
        "deltas" : [delta],
        "limits" : [upper_start+step*i]
    }
    while delta < 2.0:
        new_limit = upper_start+step*i
        insta_image = sitk.ConnectedThreshold(
            image1=base_image,
            seedList=insta_seed,
            lower=lower_limit,
            upper=new_limit,
            replaceValue=1
        )
        px_n, px_count = np.unique(sitk.GetArrayFromImage(insta_image), return_counts=True)
        delta = px_count[1] / px_count_old
        delta_data["limits"].append(new_limit)
        delta_data["deltas"].append(delta)
        px_count_old = px_count[1]
        logging.debug(f"Limit: {new_limit} delta = {delta}")
        i += 1
        

    # final segmentation
    new_limit = upper_start+step*(i-4)
    insta_image = sitk.ConnectedThreshold(
    image1=base_image,
        seedList=insta_seed,
        lower=lower_limit,
        upper=new_limit,
        replaceValue=1
    )
    uni_vals = np.unique(sitk.GetArrayFromImage(insta_image))
    logging.info(f"Unique values insta {uni_vals}")

    insta_proc_img = sitk.LabelOverlay(base_image, insta_image)
    
    fig, axs = plt.subplots()
    axs.set_title(f"Insta Image {int(file_name.split('_')[0])} limit {new_limit} last_delta {delta.round(2)}")
    axs.imshow(sitk.GetArrayViewFromImage(insta_proc_img))
    folder = "segmented_instability"
    dir_path = os.path.join(config["data_path"], folder, case)
    if os.path.exists(dir_path) is False:
        os.makedirs(dir_path)
        logging.info(f"Creating dir {folder} for case {case}") 
    if config["save_intermediate"]:
        fig.savefig(os.path.join(dir_path, file_name + ".png"))
    if config["debug"] == False:
        plt.close(fig)

    # save delta data
    fig, axs = plt.subplots()
    axs.set_title(f"Detas {file_name.split('_')[0]}")
    axs.plot(delta_data["limits"], delta_data["deltas"])
    axs.set_xlabel("upper_limits")
    axs.set_ylabel("segmented pixel growth rate")
    folder = "delta_data"
    dir_path = os.path.join(config["data_path"], folder, case)
    if os.path.exists(dir_path) is False:
        os.makedirs(dir_path)
        logging.info(f"Creating dir {folder} for case {case}")
    if config["save_intermediate"]:
        fig.savefig(os.path.join(dir_path, file_name + ".png"))
    
    if config["debug"] == False:
        plt.close(fig)

    return insta_image, status

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
    config["images"] = [1]
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
    if config["debug"] == False:
        plt.close()
    return True

def create_animation(config, case, data_folder):
    """
    Function that creates animation video for images
    """

    images = get_image_files(config, case, data_folder)
    if images == []:
        logging.warning(f"No images found to create animation for case {case}.")
        return
    animation_folder_path = os.path.join(config["results_path"], "animations", case)
    if os.path.exists(animation_folder_path) == False:
        os.makedirs(animation_folder_path)
    dat_path = os.path.join(config["data_path"], data_folder, case)
    frame = cv2.imread(os.path.join(dat_path, images[0]+ ".png"))
    height, width, layers = frame.shape
    fps = 5
    video_path = os.path.join(animation_folder_path, data_folder + ".avi")
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))
    for image in images:
        img = cv2.imread(os.path.join(dat_path, image + ".png"))
        out.write(img)
    cv2.destroyAllWindows()
    out.release()
    logging.info(f"Created {data_folder} video for case {case}")

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