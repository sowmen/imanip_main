import pandas as pd
import random
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2

from functools import partial
from glob import glob
from multiprocessing.pool import Pool
from pathlib import Path
from tqdm import tqdm


def extract(param, increment, root_dir, out_dir):
    img = cv2.imread(os.path.join(root_dir, "imd_data", param["img_path"]))
    mask = cv2.imread(
        os.path.join(root_dir, "imd_data", param["mask_path"]), cv2.IMREAD_GRAYSCALE
    )

    pixels = increment * increment
    _ratio = float(cv2.countNonZero(mask) / (mask.shape[0] * mask.shape[1])) * 100.0
    # print("Ratio printing: " + str(_ratio))
    Images = []  # Data
    Masks = []
    if _ratio >= 8.0:
        # print("Call 1")
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
                bits = cv2.countNonZero(mask_patch)

                if bits == 0 or bits == pixels:
                    continue

                ratio = float(bits / pixels) * 100.0
                if ratio > 65.0 or ratio < 5.0:
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
                    if ratio >= 65.0:
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
                    if ratio >= 65.0:
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
                    if ratio >= 65.0:
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
                    if ratio >= 65.0:
                        continue
                    image_patch2 = img[
                        lower_y : lower_y + increment, lower_x : lower_x + increment
                    ]
                    Images.append(image_patch2)
                    Masks.append(mask_patch2)

    # print("Number of patches and masks: " + str(len(Images)) + " " + str(len(Masks)))
    dir = os.path.join(
        root_dir, out_dir, param["img_path"].split("/")[-1].split(".")[0]
    )
    os.makedirs(dir, exist_ok=True)
    for i, (im, ms) in enumerate(zip(Images, Masks)):
        cv2.imwrite(os.path.join(dir, f"{i}.png"), im)
        cv2.imwrite(os.path.join(dir, f"{i}_gt.png"), ms)

    f = open(os.path.join(dir, "done.txt"), "w")
    f.close()


def extract_real(param, increment, root_dir, out_dir):
    img = cv2.imread(os.path.join(root_dir, "imd_data", param["img_path"]))

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
    if len(coords) <= 5:
        for i in coords:
            patches.append(img[i[0] : i[1], i[2] : i[3]])
    else:
        for i in range(5):
            patches.append(
                img[coords[i][0] : coords[i][1], coords[i][2] : coords[i][3]]
            )

    dir = os.path.join(
        root_dir, out_dir, param["img_path"].split("/")[-1].split(".")[0]
    )
    os.makedirs(dir, exist_ok=True)
    for i, im in enumerate(patches):
        cv2.imwrite(os.path.join(dir, f"{i}.png"), im)

    f = open(os.path.join(dir, "done.txt"), "w")
    f.close()


def main():
    patch_size = 64
    ROOT_DIR = "Image_Manipulation_Dataset/IMD2020"
    OUT_DIR = f"image_patch_{patch_size}"

    params = []
    df = pd.read_csv("imd2020.csv")
    for idx, row in df.iterrows():
        if row["label"] == 0:
            params.append({"img_path": row["image"], "mask_path": row["mask"]})
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
                    extract_real,
                    increment=patch_size,
                    root_dir=ROOT_DIR,
                    out_dir=OUT_DIR,
                ),
                params,
            ):
                pbar.update()


if __name__ == "__main__":
    main()

