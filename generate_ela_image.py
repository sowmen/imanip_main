import os
from PIL import Image
from PIL import Image, ImageChops
import pandas as pd
from tqdm import tqdm

from functools import partial
from glob import glob
from multiprocessing.pool import Pool

def ELA(param, DIR_ROOT):
    """Performs Error Level Analysis over a directory of images"""
    
    img_path = param['img_path']
    ela_path = param['ela_path']
    root_dir = param['root_dir']

    ROOT = os.path.join(DIR_ROOT, root_dir)
    
    TEMP = os.path.join(ROOT, img_path[:-4]+'ela_' + 'temp.jpg')
    SCALE = 25
    original = Image.open(os.path.join(ROOT, img_path))
    try:
        original.save(TEMP, quality=90)
        temporary = Image.open(TEMP)
        diff = ImageChops.difference(original, temporary)
        
    except:
        original.convert('RGB').save(TEMP, quality=90)
        temporary = Image.open(TEMP)
        diff = ImageChops.difference(original.convert('RGB'), temporary)
        
       
    d = diff.load()
    
    WIDTH, HEIGHT = diff.size
    for x in range(WIDTH):
        for y in range(HEIGHT):
            d[x, y] = tuple(k * SCALE for k in d[x, y])
    
    diff.save(os.path.join(ROOT,ela_path))
    os.remove(TEMP)

def main():

    df = pd.read_csv('dataset_csv/nist16_FULL.csv')
    DIR_ROOT = "Image_Manipulation_Dataset"

    params = []
    for idx, row in df.iterrows():
        params.append({"img_path": row["image_patch"], "ela_path": row["ela"], "root_dir": row["root_dir"]})
    print(f"Total: {len(params)}")

    # rows64 = df.values
    # for row in tqdm(rows64):
    #     # image, image_patch, mask_patch, label, fold, ela, root_dir = row
    #     # impath = os.path.join(image, image_patch)
    #     # ela_path = os.path.join(image, ela)
    #     image_patch, mask_patch, label, fold, ela, root_dir = row
    #     impath = image_patch
    #     ela_path = ela
    #     root = os.path.join(DIR_ROOT, root_dir)
    #     ELA((impath, ela_path), root)
    
    with Pool(processes=os.cpu_count()) as p:
        with tqdm(total=len(params)) as pbar:
            for v in p.imap_unordered(
                partial(
                    ELA,
                    DIR_ROOT=DIR_ROOT,
                ),
                params,
            ):
                pbar.update()


if __name__ == "__main__":
    main()