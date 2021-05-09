import matplotlib.pyplot as plt
import numpy as np
import cv2
import skimage
import skimage.morphology
from skimage.measure import label
import math
import pandas as pd
import re
from pathlib import Path
import imageio
import scipy as sp
import shutil
from tqdm import tqdm
import os
from PIL import Image

''' Create large context patches (with 2x propotional padding) and
move some train patches into validation dir (based on Tensorflow's split)'''

def create_im_path_to_open(path_from_manifest):
    return str(cbis_img_folder/path_from_manifest.replace('.dcm','.png')).strip()

AREA_MULTIPLIER = 2
def calculate_padded_coordinates(xmin,xmax,ymin,ymax,im_width,im_height):
    # Test Y
    #calculate_padded_coordinates(5,6,2,8,10,20)
    #calculate_padded_coordinates(5,6,10,18,10,20)
    # Test X
    #calculate_padded_coordinates(2,7,8,12,10,20)
    #calculate_padded_coordinates(4,9,8,12,10,20)

    xdiff = xmax-xmin
    ydiff = ymax-ymin
    # use same padding for both x and y so that we can obtain a square patch
    xpad = int(np.ceil(xdiff/AREA_MULTIPLIER))
    ypad = int(np.ceil(ydiff/AREA_MULTIPLIER))
    xpad = max([xpad,ypad])
    ypad = max([xpad,ypad])

    if (xmin-xpad < 0) and (xmax+xpad > im_width):
        xmin_padded = 0
        xmax_padded = im_width
    elif xmin-xpad < 0:
        # run outside boundary
        xmin_padded = 0
        xmax_padded = min([xmax+xpad-(xmin-xpad),im_width])
    elif xmax+xpad > im_width:
        xmin_padded = xmin-xpad-(xmax+xpad-im_width)
        xmax_padded = im_width
    else:
        xmin_padded = xmin-xpad
        xmax_padded = xmax+xpad
    xmin_padded = max([xmin_padded,0])
    xmax_padded = min([xmax_padded,im_width])

    if (ymin-ypad < 0) and (ymax+ypad > im_height):
        ymin_padded = 0
        ymax_padded = im_height
    elif ymin-ypad < 0:
        # run outside boundary
        ymin_padded = 0
        ymax_padded = min([ymax+ypad-(ymin-ypad),im_height])
    elif ymax+ypad > im_height:
        ymin_padded = ymin-ypad-(ymax+ypad-im_height)
        ymax_padded = im_height
    else:
        ymin_padded = ymin-ypad
        ymax_padded = ymax+ypad
    ymin_padded = max([ymin_padded,0])
    ymax_padded = min([ymax_padded,im_height])

    return xmin_padded,xmax_padded,ymin_padded,ymax_padded


### MASS TRAIN CASES  ###
df = pd.read_csv('metadata/mass_case_description_train_set.csv')
df['pathology_simplified'] = df['pathology'].str.split('_').apply(lambda x: x[0])

# what these patches will be named as, use "image file path" so it is similar with how Tensorflow names them
# for easier comparison
df["output_patch_path"] = df["image file path"].str.split('/').apply(lambda x: "-".join([
    x[1],x[2]]))
df["output_patch_path"] = df["output_patch_path"] + "-abnorm_" + df['abnormality id'].astype(str) + '.png'
df["output_patch_path"]  = df['pathology_simplified'] + "_MASS/" +  df["output_patch_path"]
df["output_patch_path"].iloc[0]

cbis_img_folder = Path("/Users/Ryan/HarvardCodes/MIT6862/cbis-ddsm-png-reorganized")
output_patch_folder = Path("/Users/Ryan/HarvardCodes/MIT6862/cbis-ddsm-large-patch/train/")

#Processing For Loop
for idx,row in df.iterrows():
    mask_path = create_im_path_to_open(row["ROI mask file path"])
    img_path = create_im_path_to_open(row["image file path"])

    image = imageio.imread(img_path)
    mask = imageio.imread(mask_path)
    if len(np.unique(mask)) > 2: # cropped and ROI mask filename was switched
        mask_path = create_im_path_to_open(row["cropped image file path"])
        mask = imageio.imread(mask_path)

    #image = process_full_mamo(image)
    im_height,im_width = image.shape # np transposed the array
    if image.shape != mask.shape:
        print("Dim not equal ",mask_path)

    # np transposed the array
    cols,rows = np.where(mask>0)
    xmin = min(rows); xmax = max(rows); ymin = min(cols); ymax = max(cols)
    input_patch = image[ymin:ymax,xmin:xmax]

    xmin_padded,xmax_padded,ymin_padded,ymax_padded = calculate_padded_coordinates(
        xmin,xmax,ymin,ymax,im_width,im_height)

    output_patch = image[ymin_padded:ymax_padded,xmin_padded:xmax_padded]

    outpath = output_patch_folder/row["output_patch_path"]
    if not outpath.parent.exists(): outpath.parent.mkdir(parents=True)
    imageio.imwrite(outpath,output_patch)

### MASS TEST CASES  ###
df = pd.read_csv('metadata/mass_case_description_test_set.csv')
df['pathology_simplified'] = df['pathology'].str.split('_').apply(lambda x: x[0])

# what these patches will be named as, use "image file path" so it is similar with how Tensorflow names them
# for easier comparison
df["output_patch_path"] = df["image file path"].str.split('/').apply(lambda x: "-".join([
    x[1],x[2]]))
df["output_patch_path"] = df["output_patch_path"] + "-abnorm_" + df['abnormality id'].astype(str) + '.png'
df["output_patch_path"]  = df['pathology_simplified'] + "_MASS/" +  df["output_patch_path"]
df["output_patch_path"].iloc[0]

cbis_img_folder = Path("/Users/Ryan/HarvardCodes/MIT6862/cbis-ddsm-png-reorganized")
output_patch_folder = Path("/Users/Ryan/HarvardCodes/MIT6862/cbis-ddsm-large-patch/test/")

#Processing For Loop
for idx,row in df.iterrows():
    mask_path = create_im_path_to_open(row["ROI mask file path"])
    img_path = create_im_path_to_open(row["image file path"])

    image = imageio.imread(img_path)
    mask = imageio.imread(mask_path)
    if len(np.unique(mask)) > 2: # cropped and ROI mask filename was switched
        mask_path = create_im_path_to_open(row["cropped image file path"])
        mask = imageio.imread(mask_path)

    #image = process_full_mamo(image)
    im_height,im_width = image.shape # np transposed the array
    if image.shape != mask.shape:
        print("Dim not equal ",mask_path)

    # np transposed the array
    cols,rows = np.where(mask>0)
    xmin = min(rows); xmax = max(rows); ymin = min(cols); ymax = max(cols)
    input_patch = image[ymin:ymax,xmin:xmax]

    xmin_padded,xmax_padded,ymin_padded,ymax_padded = calculate_padded_coordinates(
        xmin,xmax,ymin,ymax,im_width,im_height)

    output_patch = image[ymin_padded:ymax_padded,xmin_padded:xmax_padded]

    outpath = output_patch_folder/row["output_patch_path"]
    if not outpath.parent.exists(): outpath.parent.mkdir(parents=True)
    imageio.imwrite(outpath,output_patch)


### CALCIFICATION TRAIN CASES  ###
df = pd.read_csv('metadata/calc_case_description_train_set.csv')
df['pathology_simplified'] = df['pathology'].str.split('_').apply(lambda x: x[0])

# what these patches will be named as, use "image file path" so it is similar with how Tensorflow names them
# for easier comparison
df["output_patch_path"] = df["image file path"].str.split('/').apply(lambda x: "-".join([
    x[1],x[2]]))
df["output_patch_path"] = df["output_patch_path"] + "-abnorm_" + df['abnormality id'].astype(str) + '.png'
df["output_patch_path"]  = df['pathology_simplified'] + "_CALCIFICATION/" +  df["output_patch_path"]
df["output_patch_path"].iloc[0]

cbis_img_folder = Path("/Users/Ryan/HarvardCodes/MIT6862/cbis-ddsm-png-reorganized")
output_patch_folder = Path("/Users/Ryan/HarvardCodes/MIT6862/cbis-ddsm-large-patch/train/")

#Processing For Loop
for idx,row in df.iterrows():
    mask_path = create_im_path_to_open(row["ROI mask file path"])
    img_path = create_im_path_to_open(row["image file path"])

    image = imageio.imread(img_path)
    mask = imageio.imread(mask_path)
    if len(np.unique(mask)) > 2: # cropped and ROI mask filename was switched
        mask_path = create_im_path_to_open(row["cropped image file path"])
        mask = imageio.imread(mask_path)

    #image = process_full_mamo(image)
    im_height,im_width = image.shape # np transposed the array
    if image.shape != mask.shape:
        print("Dim not equal ",mask_path)

    # np transposed the array
    cols,rows = np.where(mask>0)
    xmin = min(rows); xmax = max(rows); ymin = min(cols); ymax = max(cols)
    input_patch = image[ymin:ymax,xmin:xmax]

    xmin_padded,xmax_padded,ymin_padded,ymax_padded = calculate_padded_coordinates(
        xmin,xmax,ymin,ymax,im_width,im_height)

    output_patch = image[ymin_padded:ymax_padded,xmin_padded:xmax_padded]

    outpath = output_patch_folder/row["output_patch_path"]
    if not outpath.parent.exists(): outpath.parent.mkdir(parents=True)
    imageio.imwrite(outpath,output_patch)


### CALC TEST CASES  ###
df = pd.read_csv('metadata/calc_case_description_test_set.csv')
df['pathology_simplified'] = df['pathology'].str.split('_').apply(lambda x: x[0])

# what these patches will be named as, use "image file path" so it is similar with how Tensorflow names them
# for easier comparison
df["output_patch_path"] = df["image file path"].str.split('/').apply(lambda x: "-".join([
    x[1],x[2]]))
df["output_patch_path"] = df["output_patch_path"] + "-abnorm_" + df['abnormality id'].astype(str) + '.png'
df["output_patch_path"]  = df['pathology_simplified'] + "_CALCIFICATION/" +  df["output_patch_path"]
df["output_patch_path"].iloc[0]

cbis_img_folder = Path("/Users/Ryan/HarvardCodes/MIT6862/cbis-ddsm-png-reorganized")
output_patch_folder = Path("/Users/Ryan/HarvardCodes/MIT6862/cbis-ddsm-large-patch/test/")

#Processing For Loop
for idx,row in df.iterrows():
    mask_path = create_im_path_to_open(row["ROI mask file path"])
    img_path = create_im_path_to_open(row["image file path"])

    image = imageio.imread(img_path)
    mask = imageio.imread(mask_path)
    if len(np.unique(mask)) > 2: # cropped and ROI mask filename was switched
        mask_path = create_im_path_to_open(row["cropped image file path"])
        mask = imageio.imread(mask_path)

    #image = process_full_mamo(image)
    im_height,im_width = image.shape # np transposed the array
    if image.shape != mask.shape:
        print("Dim not equal ",mask_path)

    # np transposed the array
    cols,rows = np.where(mask>0)
    xmin = min(rows); xmax = max(rows); ymin = min(cols); ymax = max(cols)
    input_patch = image[ymin:ymax,xmin:xmax]

    xmin_padded,xmax_padded,ymin_padded,ymax_padded = calculate_padded_coordinates(
        xmin,xmax,ymin,ymax,im_width,im_height)

    output_patch = image[ymin_padded:ymax_padded,xmin_padded:xmax_padded]

    outpath = output_patch_folder/row["output_patch_path"]
    if not outpath.parent.exists(): outpath.parent.mkdir(parents=True)
    imageio.imwrite(outpath,output_patch)


## MOVE APPROPRIATE TRAIN PATCHES INTO VALID SET

course_dir = Path('/Users/Ryan/HarvardCodes/MIT6862/')
src_dir = course_dir / 'cbis-ddsm-large-patch/train'
destination_dir = course_dir / 'cbis-ddsm-large-patch/valid'
label_class = "MALIGNANT_MASS"
label_classes ["MALIGNANT_MASS","MALIGNANT_CALCIFICATION","BENIGN_MASS","BENIGN_CALCIFICATION"]

for label_class in label_classes:
    valid_mm = list((course_dir / f'cbis-ddsm-patches/valid/{label_class}').glob('*'))
    # this is a Series of small context patches
    valid_mm_s = pd.Series(valid_mm)
    # # remove the patches suffix from each patch filename, we're only using the filename not the path
    valid_mm_s = valid_mm_s.apply(lambda x: Path('-'.join(x.stem.split('-')[:-1]) + x.suffix))
    valid_mm_s.name ="small_patches_name"
    valid_mm_s = valid_mm_s.drop_duplicates()
    valid_mm_s = valid_mm_s.to_frame()

    for _,s in valid_mm_s.iterrows():
        fname_to_move_to_valid = s["small_patches_name"]
        destination = destination_dir/label_class/fname_to_move_to_valid
        src = src_dir/label_class/fname_to_move_to_valid
        if not src.exists(): continue
        if not destination.parent.exists():destination.parent.mkdir(parents=True)
        print(destination)
        src.replace(destination)


## MOVE APPROPRIATE TRAIN PATCHES INTO VALID SET
course_dir = Path('/Users/Ryan/HarvardCodes/MIT6862/')
src_dir = course_dir / 'cbis-ddsm-large-patch/train'
destination_dir = course_dir / 'cbis-ddsm-large-patch/valid'
label_class = "MALIGNANT_MASS"
label_classes = ["MALIGNANT_MASS","MALIGNANT_CALCIFICATION","BENIGN_MASS","BENIGN_CALCIFICATION"]

for label_class in label_classes:
    valid_mm = list((course_dir / f'cbis-ddsm-patches/valid/{label_class}').glob('*'))
    # this is a Series of small context patches
    valid_mm_s = pd.Series(valid_mm)
    # # remove the patches suffix from each patch filename, we're only using the filename not the path
    valid_mm_s = valid_mm_s.apply(lambda x: Path('-'.join(x.stem.split('-')[:-1]) + x.suffix))
    valid_mm_s.name ="small_patches_name"
    valid_mm_s = valid_mm_s.drop_duplicates()
    valid_mm_s = valid_mm_s.to_frame()

    for _,s in valid_mm_s.iterrows():
        fname_to_move_to_valid = s["small_patches_name"]
        destination = destination_dir/label_class/fname_to_move_to_valid
        src = src_dir/label_class/fname_to_move_to_valid
        if not src.exists(): continue
        if not destination.parent.exists():destination.parent.mkdir(parents=True)
        print(destination)
        src.replace(destination)
