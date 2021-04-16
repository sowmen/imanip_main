import pandas as pd
import random
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse

from functools import partial
from glob import glob
from multiprocessing.pool import Pool
from pathlib import Path
from tqdm import tqdm

from skimage.metrics import structural_similarity as ssim


def extract_tampered(param, increment, root_dir, out_dir, **kwargs):
    img = cv2.imread(os.path.join(root_dir+kwargs['extension'], param["img_path"]))
    mask = cv2.imread(
        os.path.join(root_dir+kwargs['extension'], param["mask_path"]), cv2.IMREAD_GRAYSCALE
    )
    
    _ratio = float(cv2.countNonZero(mask) / (mask.shape[0] * mask.shape[1])) * 100.0
    # print("Ratio printing: " + str(_ratio))
    Images = []  # Data
    Masks = []
    if _ratio >= 8.0:
        d = mask.shape
        stride = int(
            increment / 2
        )  # Change kore kore dekha laagbe.. suggested: increment / 2
        for i in range(0, d[0], stride):
            for j in range(0, d[1], stride):
                x = i + increment
                y = j + increment

                if x >= d[0] or y >= d[1]:
                    break

                mask_patch = mask[i:x, j:y]

                ratio = (
                        float(cv2.countNonZero(mask_patch) / (increment * increment))
                        * 100.0
                    )
                if ratio < 13.0 or ratio >= 65.0:
                    continue
                
                valid = False
                for rows in range(mask_patch.shape[0]):
                    for cols in range(mask_patch.shape[1]):
                        if (
                            rows == 0
                            or cols == 0
                            or rows == mask_patch.shape[0] - 1
                            or cols == mask_patch.shape[1] - 1
                        ):
                            if mask_patch[rows][cols] != 0:
                                valid = True
                if valid == False:
                    continue
                img_patch = img[i:x, j:y]
                Images.append(img_patch)
                Masks.append(mask_patch)

    else:
        # less than 8%
        dims = mask.shape
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if h >= w:
                lower_y = int(y + h / 2)
                lower_x = x
                upper_y = max(0, int(lower_y - increment))
                upper_x = x
                for change in range(int(increment / 2) - 10):
                    if lower_x - 1 > 0:
                        lower_x -= 1
                        upper_x -= 1
                    else:
                        break
                while True:
                    if lower_x + increment >= dims[1]:
                        lower_x = max(0, lower_x - 1)
                    elif lower_y + increment >= dims[0]:
                        lower_y = max(0, lower_y - 1)
                    else:
                        break
                while True:
                    if upper_x + increment >= dims[1]:
                        upper_x = max(upper_x - 1, 0)
                    elif upper_y + increment >= dims[0]:
                        upper_y = max(upper_y - 1, 0)
                    else:
                        break
                mask_patch1 = mask[
                    upper_y : upper_y + increment, upper_x : upper_x + increment
                ]
                mask_patch2 = mask[
                    lower_y : lower_y + increment, lower_x : lower_x + increment
                ]
                valid1 = False
                valid2 = False
                for rows in range(mask_patch1.shape[0]):
                    for cols in range(mask_patch1.shape[1]):
                        if (
                            rows == 0
                            or cols == 0
                            or rows == mask_patch1.shape[0] - 1
                            or cols == mask_patch1.shape[1] - 1
                        ):
                            if mask_patch1[rows][cols] != 0:
                                valid1 = True
                            if mask_patch2[rows][cols] != 0:
                                valid2 = True
                if valid1:
                    ratio = (
                        float(cv2.countNonZero(mask_patch1) / (increment * increment))
                        * 100.0
                    )
                    if ratio < 13.0 or ratio >= 65.0:
                        continue
                    image_patch1 = img[
                        upper_y : upper_y + increment, upper_x : upper_x + increment
                    ]
                    Images.append(image_patch1)
                    Masks.append(mask_patch1)
                if valid2:
                    ratio = (
                        float(cv2.countNonZero(mask_patch2) / (increment * increment))
                        * 100.0
                    )
                    if ratio < 13.0 or ratio >= 65.0:
                        continue
                    image_patch2 = img[
                        lower_y : lower_y + increment, lower_x : lower_x + increment
                    ]
                    Images.append(image_patch2)
                    Masks.append(mask_patch2)
            else:
                lower_y = y
                lower_x = int(x + w / 2)
                upper_y = y
                upper_x = max(0, lower_x - increment)
                for change in range(int(increment / 2) - 10):
                    if lower_y - 1 > 0:
                        lower_y -= 1
                        upper_y -= 1
                    else:
                        break
                while True:
                    if lower_x + increment >= dims[1]:
                        lower_x = max(0, lower_x - 1)
                    elif lower_y + increment >= dims[0]:
                        lower_y = max(0, lower_y - 1)
                    else:
                        break
                while True:
                    if upper_x + increment >= dims[1]:
                        upper_x = max(upper_x - 1, 0)
                    elif upper_y + increment >= dims[0]:
                        upper_y = max(upper_y - 1, 0)
                    else:
                        break
                mask_patch1 = mask[
                    upper_y : upper_y + increment, upper_x : upper_x + increment
                ]
                mask_patch2 = mask[
                    lower_y : lower_y + increment, lower_x : lower_x + increment
                ]
                valid1 = False
                valid2 = False
                for rows in range(mask_patch1.shape[0]):
                    for cols in range(mask_patch1.shape[1]):
                        if (
                            rows == 0
                            or cols == 0
                            or rows == mask_patch1.shape[0] - 1
                            or cols == mask_patch1.shape[1] - 1
                        ):
                            if mask_patch1[rows][cols] != 0:
                                valid1 = True
                            if mask_patch2[rows][cols] != 0:
                                valid2 = True
                if valid1:
                    ratio = (
                        float(cv2.countNonZero(mask_patch1) / (increment * increment))
                        * 100.0
                    )
                    if ratio < 13.0 or ratio >= 65.0:
                        continue
                    image_patch1 = img[
                        upper_y : upper_y + increment, upper_x : upper_x + increment
                    ]
                    Images.append(image_patch1)
                    Masks.append(mask_patch1)
                if valid2:
                    ratio = (
                        float(cv2.countNonZero(mask_patch2) / (increment * increment))
                        * 100.0
                    )
                    if ratio < 13.0 or ratio >= 65.0:
                        continue
                    image_patch2 = img[
                        lower_y : lower_y + increment, lower_x : lower_x + increment
                    ]
                    Images.append(image_patch2)
                    Masks.append(mask_patch2)
                    
    patches = [(x,y) for x,y in zip(Images, Masks)]
    patches = filter_similar(patches, type="fake")
    
    
    # return patches
    dir = os.path.join(
        root_dir, out_dir, param["img_path"].split("/")[-1].split(".")[0]
    )
    os.makedirs(dir, exist_ok=True)
    
    # print("Number of patches and masks: " + str(len(Images)) + " " + str(len(Masks)) + " " + dir)
    if(len(patches) > 50):
        patches = random.sample(patches, 50)
    for i, (im, ms) in enumerate(patches):
        # print(i, (float(cv2.countNonZero(ms) / (ms.shape[0] * ms.shape[1]))* 100.0))
        cv2.imwrite(os.path.join(dir, f"{i}.png"), im)
        cv2.imwrite(os.path.join(dir, f"{i}_gt.png"), ms)
            
    f = open(os.path.join(dir, "done.txt"), "w")
    f.close()


def extract_real(param, increment, root_dir, out_dir, **kwargs):
    img = cv2.imread(os.path.join(root_dir+kwargs['extension'], param["img_path"]))

    coords = []
    d = img.shape
    patches = []  # Ei list e ekta image er jnno shobgula patch thakbe
    for i in range(0, d[0], increment):
        for j in range(0, d[1], increment):
            x = i + increment
            y = j + increment
            if x > d[0] or y > d[1]:
                break
            coords.append((i, x, j, y))

    random.shuffle(coords)
    if len(coords) <= 10:
        for i in coords:
            patches.append(img[i[0] : i[1], i[2] : i[3]])
    else:
        for i in range(10):
            patches.append(
                img[coords[i][0] : coords[i][1], coords[i][2] : coords[i][3]]
            )
    patches = filter_similar(patches, type="real")
    
    dir = os.path.join(
        root_dir, out_dir, param["img_path"].split("/")[-1].split(".")[0]
    )
    os.makedirs(dir, exist_ok=True)
    for i, im in enumerate(patches):
        cv2.imwrite(os.path.join(dir, f"{i}.png"), im)

    f = open(os.path.join(dir, "done.txt"), "w")
    f.close()


def SSIM(imageA, imageB):
    dim = (imageA.shape[0], imageA.shape[1])
    A = cv2.resize(imageA, dim, interpolation=cv2.INTER_AREA)
    B = cv2.resize(imageB, dim, interpolation=cv2.INTER_AREA)
    grayA = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(B, cv2.COLOR_BGR2GRAY)
    ans = ssim(grayA, grayB, full=True)
    ret = ans[0]
    ret += 1
    ret /= 2
    return ret


def filter_similar(patches, type):
    final_patches = []
    vis = [0 for i in range(len(patches))]
    for i in range(len(patches)):
        if vis[i]:
            continue
        vis[i] = 1
        final_patches.append(patches[i])
        for j in range(i + 1, len(patches)):
            if vis[j]:
                continue
            if type == "fake":
                if SSIM(patches[i][0], patches[j][0]) >= 0.7:
                    vis[j] = 1
            elif type == "real":
                if SSIM(patches[i], patches[j]) >= 0.7:
                    vis[j] = 1
    return final_patches


def extract_imd_orig(param, increment, root_dir, out_dir):
    # Check korish shob thik thaak ache kina
    img = cv2.imread(os.path.join(root_dir, "imd_data", param["img_path"]))
    patches = []
    d = img.shape
    for i in range(0, d[0], increment):
        for j in range(0, d[1], increment):
            x = i + increment
            y = j + increment
            if x > d[0] or y > d[1]:
                break
            patches.append(img[i:x, j:y])
    random.shuffle(patches)

    imd_patch = []
    vis = [0 for i in range(len(patches))]
    for i in range(len(patches)):
        if vis[i]:
            continue
        vis[i] = 1
        imd_patch.append(patches[i])
        for j in range(i + 1, len(patches)):
            if vis[j]:
                continue
            if SSIM(patches[i], patches[j]) >= 0.67:
                vis[j] = 1

    if len(imd_patch) > 10:
        imd_patch = random.sample(imd_patch, 10)
    # imd_patch e final patchgula ache
    dir = os.path.join(
        root_dir, out_dir, param["img_path"].split("/")[-1].split(".")[0]
    )
    os.makedirs(dir, exist_ok=True)
    for i, im in enumerate(imd_patch):
        cv2.imwrite(os.path.join(dir, f"{i}.png"), im)

    f = open(os.path.join(dir, "done.txt"), "w")
    f.close()


def main(type, patch_size):
    ROOT_DIR = "Image_Manipulation_Dataset/COCO_CMFD"
    EXTENSION = "/cmfd_full"
    OUT_DIR = f"image_patch_{patch_size}"

    if type == "real":
        label = 0
        extract_function = extract_real
    elif type == "fake":
        label = 1
        extract_function = extract_tampered
        
    params = []
    df = pd.read_csv("dataset_csv/cmfd_FULL.csv")
    for idx, row in df.iterrows():
        if row["label"] == label:
            params.append({"img_path": row["image_patch"], "mask_path": row["mask_patch"]})
    print(f"Total: {len(params)}")

    _temp = []
    for item in params:
        image_id = item["img_path"].split("/")[-1].split(".")[0]
        if os.path.exists(os.path.join(ROOT_DIR, OUT_DIR, image_id, "done.txt")):
            continue
        else:
            _temp.append((item))
    params = _temp
    print("Remaining : {}".format(len(params)))

    with Pool(processes=os.cpu_count()) as p:
        with tqdm(total=len(params)) as pbar:
            for v in p.imap_unordered(
                partial(
                    extract_function,
                    increment=patch_size,
                    root_dir=ROOT_DIR,
                    out_dir=OUT_DIR,
                    extension=EXTENSION
                ),
                params,
            ):
                pbar.update()


if __name__ == "__main__":
    main("real", 64)  

