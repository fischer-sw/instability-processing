import os
import sys
import json
import glob
import logging

from multiprocessing import Pool
from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import SimpleITK as sitk

# setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s")


def get_image_files(config, case, folder):
    """
    Function that reads image names from folder
    """
    if folder in ["raw_cases", "png_cases"]:
        dir_path = os.path.join(config["raw_data_path"])
    else:
        dir_path = os.path.join(config["data_path"])
    if os.path.exists(dir_path) is False:
        logging.error(f"Data directory {dir_path} doesn't exsist. Please check config for valid path.")
        return []

    dat_path = os.path.join(dir_path, folder, case)
    if os.path.exists(dat_path) is False:
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

    logging.info(f'Found {len(tmp_images)} images for case {case}')
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
        #create all intermediate folders
        create_intermediate_folders(config, cas)

        images = get_image_files(config, cas, "raw_cases")
        if images == []:
            images = get_image_files(config, cas, "png_cases")
        if images == []:
            logging.warning(f"No images found for case {cas}")
            continue
        images.sort()

        if config["images"] == []:
            config["debug"] = False

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
            
        create_animation(config, cas, "final_images")
        create_animation(config, cas, "png_cases")

    


def create_intermediate_folders(config, cas):
    """
    Function that creates all intermediate folders
    """
    folders = ["closed_instability", "contour", "delta_data", "final_images", "histogramms", "instability", "segmented_camera", "segmented_instability"]

    for fld in folders:

        folder_path = os.path.join(config["data_path"], fld, cas)
        if os.path.exists(folder_path) is False:
            os.makedirs(folder_path)

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
    
    new_dir_path = os.path.join(config["raw_data_path"], "png_cases", case)
    new_img_path = os.path.join(new_dir_path, file + ".png")
    img_path = os.path.join(config["raw_data_path"], "raw_cases", case, file)
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
    dir_path = os.path.join(config["raw_data_path"], folder, case)
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
    if os.path.exists(final_dir_path) is False:
        os.makedirs(final_dir_path)
    res_csv_folder = os.path.join(final_dir_path, "instabilities")
    if os.path.exists(res_csv_folder) is False:
        os.makedirs(res_csv_folder)
    res_csv_path = os.path.join(res_csv_folder ,img_name + ".png")
    if os.path.exists(res_csv_path) and config["new_files"] is False:
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
    base_array[:,300:304] = 0
    base_img = sitk.GetImageFromArray(base_array)

    cam_img = segment_camera(config, cas, base_img, img_name)

    # get cam array
    cam_array = sitk.GetArrayFromImage(cam_img)
    # replace all found cam pixel with 0 in base image
    base_array[cam_array == 1] = 0
    base_img = sitk.GetImageFromArray(base_array)
    insta_img, status = segment_instability(config, cas, base_img, img_name, cam_img)
    if status is False:
        logging.error(f"No instability found in {img_name}")
        return status

    # hull = convex_hull(config, cas, insta_img, img_name)
    new_insta_img, status = refine_instability(config, cas, insta_img, img_name)
    if status is False:
        logging.error(f"Instability found in {img_name} is more than one blob")
        return status
    
    status, final_insta_img = close_instability(config, cas, new_insta_img, img_name, cam_img)
    if status is False:
        logging.error(f"Closing instability in {img_name} failed")
        return status
    
    contours = get_contour(config, cas, final_insta_img, img_name)
    df = pd.DataFrame(contours)
    con_csv_folder = os.path.join(final_dir_path, "contours")
    if os.path.exists(con_csv_folder) is False:
        os.makedirs(con_csv_folder)
    con_path = os.path.join(con_csv_folder, img_name + ".png")
    plt.imsave(con_path, contours, cmap="Greys", dpi=1200)

    res_array = sitk.GetArrayFromImage(final_insta_img)
    final_img_path = os.path.join(config["data_path"], "final_images", cas)
    if os.path.exists(final_img_path) is False:
        os.makedirs(final_img_path)
    
    plt.imsave(res_csv_path, res_array, cmap="Greys", dpi=1200)

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
    raw_img_path = os.path.join(config["raw_data_path"], "png_cases", case, file_name + ".png")
    if os.path.exists(raw_img_path):
        reader = sitk.ImageFileReader()
        reader.SetImageIO("PNGImageIO")
        reader.SetFileName(raw_img_path)
        raw_image = reader.Execute()
    else:
        raw_img_path = os.path.join(config["raw_data_path"], "raw_cases", case, file_name + ".tiff")
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
    if config["debug"] is False:
        plt.close(fig)

    folder = "contour"
    dir_path = os.path.join(config["data_path"], folder, case)
    if os.path.exists(dir_path) is False:
        os.makedirs(dir_path)
        logging.info(f"Creating dir {folder} for case {case}")
    if config["save_intermediate"]:
        fig.savefig(os.path.join(dir_path, file_name + ".png"))
    
    return contour_image

def close_instability(config, case, base_image, file_name, cam_image):
    """
    Function that trys to close camera hole. WORK IN PROGRESS!
    """

    status = True
    image_array = sitk.GetArrayFromImage(base_image)
    lower = (0,0)
    upper = (0,0)
    rates = {
        "line" : [],
        "values" : []
    }
    contours, hierarchy = cv2.findContours(image=image_array, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    cont = contours[0]

    M = cv2.moments(cont)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    cX -= 10
    logging.info(f"cX: {cY}, cY: {cX}")

    contour_image = cv2.drawContours(image=np.zeros(image_array.shape), contours=contours, contourIdx=0, color=(1, 0, 0), thickness=7, lineType=cv2.LINE_AA)

    bound_rect = cv2.boundingRect(cont)
    logging.info(bound_rect)

    tl_x = bound_rect[0]
    tl_y = bound_rect[1]
    width = bound_rect[2]
    height = bound_rect[3]

    #get array from camera image
    cam_array = sitk.GetArrayFromImage(cam_image)

    step = 75
    counter = 1
    m = 90

    while abs(m) > 5 and counter < 11:
        
        cam_array = sitk.GetArrayFromImage(cam_image)
        cam_array[:,:tl_x-counter*2*step] = 0 
        # cam_array[:,tl_x+counter*step:] = 0
        cam_array[:,tl_x:] = 0

        y,x = np.where(cam_array == 1)
        coef = np.polyfit(x,y,1)
        m = coef[0]
        poly1d_fn = np.poly1d(coef)

        fig, axs = plt.subplots()
        axs.set_title(f"Test instability {counter}")
        axs.axline((0, poly1d_fn(0)), slope=m)
        pos = axs.imshow(cam_array, cmap="Greys")
        fig.colorbar(pos, ax=axs)
        if config["debug"] is False:
            plt.close(fig)

        counter += 1        
    logging.info(f"Slope: {m}")

    if abs(m) > 5:
        logging.error(f"Camera segmentation near instability bounding box failed. Skiping {file_name}")
        status = False
        return status, base_image

    bb_div_y = poly1d_fn(tl_x)
    upper_x = tl_x + width
    upper_y = tl_y
    for i in range(int(tl_y), int(bb_div_y)):
        tmp_line = image_array[i,:]
        test = np.where(tmp_line == 1)[0][0]
        if test < upper_x:
            upper_x = test
            upper_y = i

    upper = (upper_x, upper_y)

    lower_x = tl_x + width
    lower_y = tl_y + height

    for i in range(int(bb_div_y), int(tl_y + height)):
        tmp_line = image_array[i,:]
        test = np.where(tmp_line == 1)[0][0]
        if test < lower_x:
            lower_x = test
            lower_y = i

    lower = (lower_x, lower_y)

    fig, axs = plt.subplots()
    axs.set_title(f"Contours + Center")
    axs.plot([cX],[cY],markersize=10, marker='*', color="red"),
    axs.plot([upper_x, lower_x],[upper_y, lower_y],markersize=8, marker='*', color="green"),
    axs.axline((0, poly1d_fn(0)), slope=m)
    axs.plot([tl_x, tl_x+ width, tl_x+ width, tl_x, tl_x], [tl_y, tl_y, tl_y +height, tl_y + height, tl_y])
    axs.plot([tl_x], [bb_div_y], marker="*", color="orange", markersize=10)
    # axs.legend() 
    pos = axs.imshow(contour_image, cmap="Greys")
    fig.colorbar(pos, ax=axs)

    folder = "closed_instability"
    dir_path = os.path.join(config["data_path"], folder, case)
    if os.path.exists(dir_path) is False:
        os.makedirs(dir_path)
        logging.info(f"Creating dir {folder} for case {case}")
    if config["save_intermediate"]:
        fig.savefig(os.path.join(dir_path, file_name + ".png"))
    if config["debug"] is False:
        plt.close(fig)

    if upper != (0,0) and lower != (0,0):
        image_array = cv2.line(image_array, lower, upper, color=(1,0,0), thickness=5)
    
    tmp_image = sitk.GetImageFromArray(image_array)
    cam_seed = [(cX,cY)]
    px_val = tmp_image.GetPixel(cam_seed[0])
    logging.info(f"Seed value: {px_val}")
    if px_val == 1:
        cam_seed = [(cX, int(poly1d_fn(cX)))]
        px_val = tmp_image.GetPixel(cam_seed[0])
        logging.info(f"New seed value: {px_val}")
    

    fig, axs = plt.subplots()
    axs.set_title(f"Closed instability")
    axs.plot([cX],[cY],markersize=10, marker='*', color="red"),
    axs.plot([cX],[poly1d_fn(cX)],markersize=10, marker='*', color="red"),
    # axs.plot([upper_x, lower_x],[upper_y, lower_y],markersize=10, marker='.', color="green")
    pos = axs.imshow(image_array, cmap="Greys")
    fig.colorbar(pos, ax=axs)
    if config["debug"] is False:
        plt.close(fig)

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
    if config["debug"] is False:
        plt.close(fig)
    
    return status, sitk.GetImageFromArray(image_array)

def refine_instability(config, case, base_image, file_name):
    """
    Function that fills holes in instability segmentation
    """
    tmp_image = base_image
    status = True

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
    
    final_array = sitk.GetArrayViewFromImage(tmp_image)
    contours, hierarchy = cv2.findContours(image=final_array, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    tmp_array = final_array.copy()

    fig, axs = plt.subplots()
    axs.set_title(f"Contour Image after dil/er {file_name.split('_')[0]}")
    pos = axs.imshow(final_array, cmap="Greys")
    fig.colorbar(pos, ax=axs)
    if config["debug"] is False:
        plt.close(fig)

    if len(contours) != 1:

        length = 0
        rm_contours = [ele for ele in contours]
        cont_len = [len(ele) for ele in contours]
        for ele in contours:
            if len(ele) > length:
                contours = [ele]
                length = len(ele)
            
        rm_contours.remove(ele)

        for ele in rm_contours:
            bound_rect = cv2.boundingRect(ele)
            logging.debug(f"Remove {bound_rect}")

            tl_x = bound_rect[0]
            tl_y = bound_rect[1]
            width = bound_rect[2]
            height = bound_rect[3]

            tmp_array[tl_y:tl_y+height,tl_x:tl_x+width] = 0
                 
    final_array = tmp_array
    logging.info(f"Contours_n: {len(contours)}")
    if len(contours) != 1:
        contour_image = cv2.drawContours(image=np.zeros(final_array.shape), contours=contours, contourIdx=-1, color=(1, 0, 0), thickness=7, lineType=cv2.LINE_AA)
        fig, axs = plt.subplots()
        axs.set_title(f"Contour Image {file_name.split('_')[0]}")
        pos = axs.imshow(contour_image, cmap="Greys")
        fig.colorbar(pos, ax=axs)
        if config["debug"] is False:
            plt.close(fig)
        status = False
        return base_image, status

    fig, axs = plt.subplots()
    axs.set_title(f"Instability after Dilate/Erode")
    pos = axs.imshow(final_array, cmap="Greys")
    fig.colorbar(pos, ax=axs)
    
    folder = "instability"
    dir_path = os.path.join(config["data_path"], folder, case)
    if os.path.exists(dir_path) is False:
        os.makedirs(dir_path)
        logging.info(f"Creating dir {folder} for case {case}")
    if config["save_intermediate"]:
        fig.savefig(os.path.join(dir_path, file_name + ".png"))
    if config["debug"] is False:
        plt.close(fig)
    tmp_image = sitk.GetImageFromArray(final_array)    
    return tmp_image, status

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
    if config["debug"] is False:
        plt.close(fig)

    return chull


def segment_camera(config, case, base_image, file_name):
    """
    Function that segements instability from an image
    """
    cam_seed = [(10,10)]
    px_val = base_image.GetPixel(cam_seed[0])

    insta_seed = [(1450,1200)]
    insta_px_val = base_image.GetPixel(insta_seed[0])
    logging.info(f"Insta seed value: {insta_px_val}")

    fig, axs = plt.subplots()
    axs.set_title(f"Base Image {file_name.split('_')[0]}")
    axs.imshow(sitk.GetArrayViewFromImage(base_image), cmap="gray")
    if config["debug"] is False:
        plt.close(fig)

    upper_start = 45

     # initial segmentation
    cam_image = sitk.ConnectedThreshold(
            image1=base_image,
            seedList=cam_seed,
            lower=0,
            upper=upper_start,
            replaceValue=1
        )

    px_n, px_count = np.unique(sitk.GetArrayFromImage(cam_image), return_counts=True)

    if len(px_n) == 1:
        return base_image, False
    
    px_count_old = px_count[1]
    delta = px_count[1] / px_count_old
    i = 0
    step = 1
    tmp_step = 0
    step_thresh = 0.5
    new_limit = upper_start
    delta_data = {
        "deltas" : [delta],
        "limits" : [upper_start+step*i]
    }

    cam_array = sitk.GetArrayFromImage(cam_image)
    # get corner values to check if segementation has run to boundary
    px_offset = 50
    lr_cor_val = cam_image.GetPixel(cam_array.shape[1]- px_offset, cam_array.shape[0]- px_offset)
    ur_cor_val = cam_image.GetPixel(cam_array.shape[1]- px_offset, px_offset)


    while tmp_step < step_thresh and (lr_cor_val == 0 or ur_cor_val == 0):
        new_limit = upper_start+step*i

        lr_cor_val = cam_image.GetPixel(cam_array.shape[1]- px_offset, cam_array.shape[0]- px_offset)
        ur_cor_val = cam_image.GetPixel(cam_array.shape[1]- px_offset, px_offset)

        cam_image = sitk.ConnectedThreshold(
            image1=base_image,
            seedList=cam_seed,
            lower=0,
            upper=new_limit,
            replaceValue=1
        )
        px_n, px_count = np.unique(sitk.GetArrayFromImage(cam_image), return_counts=True)
        if len(px_n) == 1:
            return base_image, False
        delta = px_count[1] / px_count_old
        tmp_step = abs(delta_data["deltas"][-1] - delta)
        delta_data["limits"].append(new_limit)
        delta_data["deltas"].append(delta)

        # if delta > 1.1:
        #     insta_proc_img = sitk.LabelOverlay(base_image, cam_image)
        #     fig, axs = plt.subplots()
        #     axs.set_title(f"Cam Image {int(file_name.split('_')[0])} limit {new_limit} tmp_step {tmp_step.round(2)}")
        #     axs.imshow(sitk.GetArrayViewFromImage(insta_proc_img))
        #     if config["debug"] is False:
        #         plt.close(fig)

        px_count_old = px_count[1]
        if tmp_step > step_thresh and (lr_cor_val == 1 or ur_cor_val == 1):
            logging.info(f"Limit: {new_limit} delta = {delta}")
            insta_proc_img = sitk.LabelOverlay(base_image, cam_image)
            fig, axs = plt.subplots()
            axs.set_title(f"Cam Image {int(file_name.split('_')[0])} limit {new_limit} tmp_step {tmp_step.round(2)}")
            axs.imshow(sitk.GetArrayViewFromImage(insta_proc_img))
            if config["debug"] is False:
                plt.close(fig)
        i += 1        

    # final segmentation
    tmp_delta = pd.DataFrame(delta_data)
    max_pos = tmp_delta.query(f"limits < 70")["deltas"].idxmax()
    new_limit = int(tmp_delta["limits"][max_pos])
    logging.info(f"Cam final upper value {new_limit}")
    cam_image = sitk.ConnectedThreshold(
    image1=base_image,
        seedList=cam_seed,
        lower=0,
        upper=new_limit,
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
    if config["debug"] is False:
        plt.close(fig)

    # save delta data
    fig, axs = plt.subplots()
    axs.set_title(f"Cam Deltas {file_name.split('_')[0]}")
    axs.plot(delta_data["limits"], delta_data["deltas"])
    axs.set_xlabel("upper_limits")
    axs.set_ylabel("segmented pixel growth rate")
    folder = "cam_delta_data"
    dir_path = os.path.join(config["data_path"], folder, case)
    if os.path.exists(dir_path) is False:
        os.makedirs(dir_path)
        logging.info(f"Creating dir {folder} for case {case}")
    if config["save_intermediate"]:
        fig.savefig(os.path.join(dir_path, file_name + ".png"))
    
    if config["debug"] is False:
        plt.close(fig)

    return cam_image

def segment_instability(config, case, base_image, file_name, cam_seg):
    """
    Function that segements camera from base image
    """

    

    tmp_image = sitk.BinaryErode(
        image1=cam_seg,
        backgroundValue=0.0,
        foregroundValue=1.0,
        boundaryToForeground=True,
        kernelRadius=(1,20)
    )

    tmp_image = sitk.BinaryDilate(
        image1=tmp_image,
        backgroundValue=0.0,
        foregroundValue=1.0,
        boundaryToForeground=True,
        kernelRadius=(1,20)
    )


    cam_array = sitk.GetArrayFromImage(tmp_image)
    boundary = 400
    max_x = cam_array.shape[0]
    max_y = cam_array.shape[1]
    # set boundary values to 0 to get rid of artifacts
    # cam_array[:boundary + max_x-boundary:,:boundary + max_y - boundary:] = 0
    cam_array[:boundary, :] = 0
    cam_array[:, :boundary] = 0
    cam_array[max_x-boundary:,:] = 0
    cam_array[:, max_y - boundary:] = 0
    # get bounding box of camera
        # get contours
    contours, hierarchy = cv2.findContours(image=cam_array, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    cont = contours[0]
    bound_rect = cv2.boundingRect(cont)
    logging.info(bound_rect)

    tl_x = bound_rect[0]
    tl_y = bound_rect[1]
    width = bound_rect[2]
    height = bound_rect[3]

    # get seeds for instabiltiy segmentation around camera

    # get axes of camera
    y,x = np.where(cam_array == 1)
    coef = np.polyfit(x,y,1)
    m = coef[0]
    poly1d_fn = np.poly1d(coef)

    seeds = []
    offset = 20
    seeds.append((tl_x + width + offset, int(poly1d_fn(tl_x + width + offset))))
    seeds.append((tl_x + 50, int(poly1d_fn(tl_x + 50) - height/2 - offset)))
    seeds.append((tl_x + 50, int(poly1d_fn(tl_x + 50) + height/2 + offset)))
    seeds.append((tl_x + 150, int(poly1d_fn(tl_x + 150) - height/2 - offset)))
    seeds.append((tl_x + 150, int(poly1d_fn(tl_x + 150) + height/2 + offset)))
    seeds.append((tl_x + 225, int(poly1d_fn(tl_x + 225) - height/2 - offset)))
    seeds.append((tl_x + 225, int(poly1d_fn(tl_x + 225) + height/2 + offset)))
    logging.info(f"Insta Seeds : {seeds}")
    fig, axs = plt.subplots()
    axs.set_title("Cam position")
    for ele in seeds:
        axs.plot([ele[0]], [ele[1]], marker="*", markersize=8, color="blue")
    axs.axline((0, poly1d_fn(0)), slope=m)
    axs.plot([tl_x, tl_x+ width, tl_x+ width, tl_x, tl_x], [tl_y, tl_y, tl_y +height, tl_y + height, tl_y])
    pos = axs.imshow(cam_array, cmap="Greys")
    fig.colorbar(pos, ax=axs)
    if config["debug"] is False:
        plt.close(fig)


    status = True
    # insta_seed = [(1450,1100), (1200,975), (1200,1175), (1350,975), (1350,1175)]
    # insta_seed = [(1450,1100), (1200,975)]
    insta_seed = seeds

    px_vals = np.array([base_image.GetPixel(ele) for ele in insta_seed])
    px_val = px_vals[(px_vals > 0) & (px_vals < 230)]
    if px_val.size == 0:
        return base_image, False
    else:
        px_val = int(px_val.mean())
    logging.info(f"Insta seed value: {px_val}")
    lower_limit = 1
    if px_val > 230:
        return base_image, False
    else:
        upper_start = px_val + 5
    # initial segmentation
    insta_image = sitk.ConnectedThreshold(
            image1=base_image,
            seedList=insta_seed,
            lower=lower_limit,
            upper=upper_start,
            replaceValue=1
        )
    insta_array = sitk.GetArrayFromImage(insta_image)
    px_n, px_count = np.unique(insta_array, return_counts=True)

    if len(px_n) == 1:
        return base_image, False
    
    px_count_old = px_count[1]
    delta = px_count[1] / px_count_old
    i = 0
    step = 5
    tmp_step = 0
    step_thresh = 0.5
    new_limit = upper_start
    delta_data = {
        "deltas" : [delta],
        "limits" : [upper_start+step*i]
    }

    # get corner values to check if segementation has run to boundary
    px_offset = 50
    lr_cor_val = insta_image.GetPixel(insta_array.shape[1]- px_offset, insta_array.shape[0]- px_offset)
    ur_cor_val = insta_image.GetPixel(insta_array.shape[1]- px_offset, px_offset)

    while tmp_step < step_thresh and (lr_cor_val == 0 or ur_cor_val == 0):
        new_limit = upper_start+step*i
        lr_cor_val = insta_image.GetPixel(insta_array.shape[1]- px_offset, insta_array.shape[0]- px_offset)
        ur_cor_val = insta_image.GetPixel(insta_array.shape[1]- px_offset, px_offset)

        insta_image = sitk.ConnectedThreshold(
            image1=base_image,
            seedList=insta_seed,
            lower=lower_limit,
            upper=new_limit,
            replaceValue=1
        )
        px_n, px_count = np.unique(sitk.GetArrayFromImage(insta_image), return_counts=True)
        if len(px_n) == 1:
            return base_image, False
        delta = px_count[1] / px_count_old
        tmp_step = abs(delta_data["deltas"][-1] - delta)
        delta_data["limits"].append(new_limit)
        delta_data["deltas"].append(delta)
        px_count_old = px_count[1]
        if tmp_step > step_thresh and (lr_cor_val == 1 or ur_cor_val == 1):
            logging.info(f"Limit: {new_limit} delta = {delta}")
            insta_proc_img = sitk.LabelOverlay(base_image, insta_image)
            fig, axs = plt.subplots()
            axs.set_title(f"Insta Image {int(file_name.split('_')[0])} limit {new_limit} last_delta {delta.round(2)}")
            axs.imshow(sitk.GetArrayViewFromImage(insta_proc_img))
            
            if config["debug"] is False:
                plt.close(fig)
        i += 1
        
        # get image for every iteration
         
        # insta_proc_img = sitk.LabelOverlay(base_image, insta_image)
        # fig, axs = plt.subplots()
        # axs.set_title(f"Insta Image {int(file_name.split('_')[0])} limit {new_limit} last_delta {delta.round(2)}")
        # axs.imshow(sitk.GetArrayViewFromImage(insta_proc_img))
        # if config["debug"] is False:
        #     plt.close(fig)
        

    # final segmentation
    new_limit = upper_start+step*(i-4)
    insta_image = sitk.ConnectedThreshold(
    image1=base_image,
        seedList=insta_seed,
        lower=lower_limit,
        upper=new_limit,
        replaceValue=1
    )
    seg_array = sitk.GetArrayFromImage(insta_image)
    uni_vals, uni_counts = np.unique(seg_array, return_counts=True)

    if len(uni_vals) > 1 and uni_counts[-1] < 100:
        return base_image, False 
    logging.info(f"Unique values insta {uni_vals}, {uni_counts}")

    insta_proc_img = sitk.LabelOverlay(base_image, insta_image)
    
    fig, axs = plt.subplots()
    axs.set_title(f"Insta Image {int(file_name.split('_')[0])} limit {new_limit} last_delta {delta.round(2)}")
    for ele in insta_seed:
        axs.plot([ele[0]], [ele[1]], marker="*", markersize=8, color="blue")
    axs.imshow(sitk.GetArrayViewFromImage(insta_proc_img))
    folder = "segmented_instability"
    dir_path = os.path.join(config["data_path"], folder, case)
    if os.path.exists(dir_path) is False:
        os.makedirs(dir_path)
        logging.info(f"Creating dir {folder} for case {case}") 
    if config["save_intermediate"]:
        fig.savefig(os.path.join(dir_path, file_name + ".png"))
    if config["debug"] is False:
        plt.close(fig)

    # save delta data
    fig, axs = plt.subplots()
    axs.set_title(f"Insta Deltas {file_name.split('_')[0]}")
    axs.plot(delta_data["limits"], delta_data["deltas"])
    axs.set_xlabel("upper_limits")
    axs.set_ylabel("segmented pixel growth rate")
    folder = "insta_delta_data"
    dir_path = os.path.join(config["data_path"], folder, case)
    if os.path.exists(dir_path) is False:
        os.makedirs(dir_path)
        logging.info(f"Creating dir {folder} for case {case}")
    if config["save_intermediate"]:
        fig.savefig(os.path.join(dir_path, file_name + ".png"))
    
    if config["debug"] is False:
        plt.close(fig)

    return insta_image, status

def make_histo(config, case, folder, name) -> bool:
    """
    Function that creates histogram from image values
    """
    
    base_path=os.path.join(config["data_path"], "histogramms", folder, case)
    if os.path.exists(base_path) is False:
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
    if config["debug"] is False:
        plt.close()
    return True

def remove_empty_folders(path_abs):
    walk = list(os.walk(path_abs))
    for path, _, _ in walk[::-1]:
        if len(os.listdir(path)) == 0:
            os.rmdir(path)
            logging.info(f"Removed {path}")

def create_animation(config, case, data_folder):
    """
    Function that creates animation video for images
    """
    
    images = get_image_files(config, case, data_folder)
    if images == []:
        logging.warning(f"No images found to create animation for case {case}.")
        return
    animation_folder_path = os.path.join(config["results_path"], "animations", case)
    if os.path.exists(animation_folder_path) is False:
        os.makedirs(animation_folder_path)

    if data_folder == "png_cases":
        dat_path = os.path.join(config["raw_data_path"], data_folder, case)
    else:
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

if __name__ == "__main__":
    calc_case_ratio()
    remove_empty_folders(os.path.join(sys.path[0], "data"))
    remove_empty_folders(os.path.join(sys.path[0], "results"))
