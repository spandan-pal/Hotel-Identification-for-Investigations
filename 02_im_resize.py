import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2

summary = pd.read_csv("Support Files/summary.csv")
summary = summary[:30]

for hotel in (pbar:= tqdm(summary["Hotel"])):
    path = "Data_Raw/train_images/" + str(hotel)

    images = os.listdir(path)
    images = [image for image in images if str(image)[0] != "." ]
    
    try:
        os.mkdir("IM_Resize")
    except FileExistsError:
        pass

    try:
        os.mkdir("IM_Resize/images")
    except FileExistsError:
        pass

    try:
        dir_path = "IM_Resize/images/" + str(hotel)
        os.mkdir(dir_path)
    except FileExistsError:
        pass

    for image in images:
        pbar.set_description(f"Converting images for {hotel} - {image}")
        im_path = path + "/" + str(image)
        write_path = "IM_Resize/images/" + str(hotel) + "/" + str(image)
        image = cv2.imread(im_path)
        new_image = cv2.resize(image, (256, 256))
        cv2.imwrite(write_path, new_image)

path = "Data_Raw/train_masks/"

images = os.listdir(path)
images = [image for image in images if str(image)[0] != "." ]

for image in (pbar:= tqdm(images)):  
    try:
        os.mkdir("IM_Resize")
    except FileExistsError:
        pass

    try:
        os.mkdir("IM_Resize/masks")
    except FileExistsError:
        pass
    
    pbar.set_description(f"Converting image {image}")
    im_path = path + "/" + str(image)
    write_path = "IM_Resize/masks/" + str(image)
    image = cv2.imread(im_path)
    new_image = cv2.resize(image, (256, 256))
    cv2.imwrite(write_path, new_image)