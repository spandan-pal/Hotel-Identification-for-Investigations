import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def load_and_invert_mask(mask_path):
    mask = Image.open(mask_path).convert("L")
    mask_array = np.array(mask)
    
    inverted_mask = 255 - mask_array

    for i in range(len(inverted_mask)):
        for j in range(len(inverted_mask[i])):
            if inverted_mask[i][j] != 255:
                inverted_mask[i][j] = 0
    
    return inverted_mask

def apply_mask(image_path, mask):
    image = Image.open(image_path)
    image = image.convert("RGBA")

    alpha = np.full((image.height, image.width), 255, dtype=np.uint8)
    alpha[mask == 0] = 0

    result = Image.merge("RGBA", (image.split()[:3] + (Image.fromarray(alpha),)))

    return result

def get_mask(image):
    if type(image) == np.ndarray:
        im_array = image
    else:
        im_array = np.array(image)

    i_s = {}
    count = 0
    for i in range(len(im_array)):
        start = False
        j_s = []
        for j in range(len(im_array[i])):
            if im_array[i][j][3] == 0:
                start = True
                count += 1
                j_s.append(j)
        if start:
            i_s[i] = j_s
    
    keys = list(i_s.keys())
    if not keys:
        return ((0, 0), 0, 0, im_array)
    else:
        start_y = keys[0]
        start_x = i_s[start_y][0]
        height = (keys[-1] - keys[0] + 1)
        width = (i_s[start_y][-1] - i_s[start_y][0] + 1)

    return ((start_y, start_x), height, width, im_array)

def blend_horizontal(image, mid=False):
    start_pixel, height, width, im_array = get_mask(image)
    start_pixel_y, start_pixel_x = start_pixel

    while width != 0:
        px_left = start_pixel_x
        px_right = im_array.shape[1] - (start_pixel_x + width)

        if ((px_left - px_right) > 150 or mid and not ((px_right - px_left > 150))):
            cut_x = max(0, start_pixel_x - width)

            flip = np.flip(im_array[start_pixel_y: start_pixel_y + height, cut_x: start_pixel_x].copy(), axis=1)
            im_array[start_pixel_y: start_pixel_y + height, start_pixel_x: start_pixel_x + start_pixel_x - cut_x] = flip

            start_pixel, height, width, im_array = get_mask(im_array)
            start_pixel_y, start_pixel_x = start_pixel
        
        elif ((px_right - px_left > 150) or mid) and not ((px_left - px_right) > 150):
            cut_x = min(im_array.shape[1], start_pixel_x + 2*width)

            flip = np.flip(im_array[start_pixel_y: start_pixel_y + height, start_pixel_x + width: cut_x].copy(), axis=1)
            im_array[start_pixel_y: start_pixel_y + height, 2*(start_pixel_x + width) - cut_x: start_pixel_x + width] = flip

            start_pixel, height, width, im_array = get_mask(im_array)
            start_pixel_y, start_pixel_x = start_pixel
        else:
            im_array[:, :start_pixel_x + int(width/2) + 2] = blend_horizontal(im_array[:, :start_pixel_x + int(width/2) + 2], mid=True)
            im_array[:, start_pixel_x + int(width/2):] = blend_horizontal(im_array[:, start_pixel_x + int(width/2):], mid=True)
            width = 0

    return im_array

def blend_vertical(image, mid=False):
    start_pixel, height, width, im_array = get_mask(image)
    start_pixel_y, start_pixel_x = start_pixel

    while width != 0:
        px_up = start_pixel_y
        px_down = im_array.shape[0] - (start_pixel_y + height)

        if ((px_up - px_down) > 150 or mid and not ((px_down - px_up > 150))):
            cut_y = max(0, start_pixel_y - height)

            flip = np.flip(im_array[cut_y: start_pixel_y, start_pixel_x: start_pixel_x + width].copy(), axis=0)
            im_array[start_pixel_y: start_pixel_y + start_pixel_y - cut_y, start_pixel_x: start_pixel_x + width] = flip

            start_pixel, height, width, im_array = get_mask(im_array)
            start_pixel_y, start_pixel_x = start_pixel
        
        elif ((px_down - px_up > 150) or mid) and not ((px_up - px_down) > 150):
            cut_y = min(im_array.shape[0], start_pixel_y + 2*height)

            flip = np.flip(im_array[start_pixel_y + height: cut_y, start_pixel_x: start_pixel_x + width].copy(), axis=0)
            im_array[2*(start_pixel_y + height) - cut_y: start_pixel_y + height, start_pixel_x: start_pixel_x + width] = flip

            start_pixel, height, width, im_array = get_mask(im_array)
            start_pixel_y, start_pixel_x = start_pixel
        else:
            im_array[:start_pixel_y + int(height/2) + 2, :] = blend_vertical(im_array[:start_pixel_y + int(height/2) + 2, :], mid=True)
            im_array[start_pixel_y + int(height/2):, :] = blend_vertical(im_array[start_pixel_y + int(height/2):, :], mid=True)
            width = 0

    return im_array

try:
    os.mkdir("Support Files")
except FileExistsError:
    pass

try:
    os.mkdir("Support Files/Masking Summary")
except FileExistsError:
    pass

try:
    os.mkdir("Support Files/Image Generation Summary")
except FileExistsError:
    pass

mask_img_list = os.listdir("IM_Resize/masks/")
mask_img_list = [image for image in mask_img_list if str(image)[0] != "." ]

hotels = os.listdir("IM_Resize/images/")
hotels = [10129] #Change the number

test_split = 0.2
val_split = 0.15
per_hotel_data = 10000
for hotel in tqdm(hotels):
    df = []
    images = os.listdir("IM_Resize/images/" + str(hotel) + "/")
    images = [image for image in images if str(image)[0] != "." ]
    
    try:
        df = pd.read_csv(f"Support Files/Masking Summary/masking_summary{hotel}.csv")
    except FileNotFoundError:
        for image in images:
            for mask_img in mask_img_list:
                df.append([hotel, image, mask_img])
        
        df = pd.DataFrame(df, columns=['Hotel', 'Image', 'Mask_Image'])
        df["Combined"] = df["Hotel"].astype(str) + "/" + df["Image"] + "/" + df["Mask_Image"]
        df.to_csv(f"Support Files/Masking Summary/masking_summary{hotel}.csv", index=False)

    hotel = list(df["Hotel"].unique())[0]
    hotel_images = list(df.loc[df["Hotel"].astype(str) == str(hotel)]['Combined'])

    track = []
    test_size = int(per_hotel_data * test_split)
    train_size = per_hotel_data - test_size
    val_size = train_size * val_split
    train_size = train_size - val_size

    try:
        processed_imgs = np.loadtxt(f"Support Files/Image Generation Summary/{hotel}_train.txt", dtype="str")
        track_train = list(processed_imgs)
        track = track + track_train
        train_size = train_size - len(processed_imgs) + 1
    except FileNotFoundError:
        track_train = [f"{hotel}_train"]

    try:
        processed_imgs = np.loadtxt(f"Support Files/Image Generation Summary/{hotel}_test.txt", dtype="str")
        track_test = list(processed_imgs)
        track = track + track_test
        test_size = test_size - len(processed_imgs) + 1
    except FileNotFoundError:
        track_test = [f'{hotel}_test']

    try:
        processed_imgs = np.loadtxt(f"Support Files/Image Generation Summary/{hotel}_val.txt", dtype="str")
        track_val = list(processed_imgs)
        track = track + track_val
        val_size = val_size - len(processed_imgs) + 1
    except FileNotFoundError:
        track_val = [f'{hotel}_val']
    
    hotel_images = list(set(hotel_images) - set(track))
    test_images = np.random.choice(hotel_images, int(test_size), replace=False)
    hotel_images = list(set(hotel_images) - set(test_images))
    train_images = np.random.choice(hotel_images, int(train_size) + int(val_size), replace=False)

    val_images = np.random.choice(train_images, int(val_size), replace=False)
    train_images = list(set(train_images) - set(val_images))

    count = len(track_train) - 1
    for c_image in tqdm(train_images):
        track_train.append(c_image)
        c_image = c_image.split("/")
        hotel = c_image[0]
        h_image = c_image[1]
        mask = c_image[2]

        try:
            os.mkdir("HotelTraffickingData")
        except FileExistsError:
            pass

        try:
            os.mkdir("HotelTraffickingData/horizaontal")
        except FileExistsError:
            pass
        
        try:
            os.mkdir("HotelTraffickingData/vertical")
        except FileExistsError:
            pass
        
        try:
            os.mkdir("HotelTraffickingData/horizaontal/train")
        except FileExistsError:
            pass

        try:
            os.mkdir("HotelTraffickingData/vertical/train")
        except FileExistsError:
            pass

        try:
            os.mkdir("HotelTraffickingData/horizaontal/train/" + str(hotel))
        except FileExistsError:
            pass

        try:
            os.mkdir("HotelTraffickingData/vertical/train/" + str(hotel))
        except FileExistsError:
            pass
        
        mask_path = "IM_Resize/masks/" + str(mask)
        image_path = "IM_Resize/images/" + str(hotel) + "/" + str(h_image)
        write_path_horizontal = "HotelTraffickingData/horizaontal/train/" + str(hotel) + str(f"/{count}.png")
        write_path_vertical = "HotelTraffickingData/vertical/train/" + str(hotel) + str(f"/{count}.png")
        
        loaded_mask = load_and_invert_mask(mask_path)
        masked_image = apply_mask(image_path, loaded_mask)

        im_array = blend_horizontal(masked_image)
        im = Image.fromarray(im_array)
        im.save(write_path_horizontal)

        im_array = blend_vertical(masked_image)
        im = Image.fromarray(im_array)
        im.save(write_path_vertical)

        with open(f"Support Files/Image Generation Summary/{hotel}_train.txt", "w") as fp:
            fp.write("\n".join(track_train))

        count += 1
    
    count = len(track_val) - 1
    for c_image in tqdm(val_images):
        track_val.append(c_image)
        c_image = c_image.split("/")
        hotel = c_image[0]
        h_image = c_image[1]
        mask = c_image[2]

        try:
            os.mkdir("HotelTraffickingData")
        except FileExistsError:
            pass

        try:
            os.mkdir("HotelTraffickingData/horizaontal")
        except FileExistsError:
            pass
        
        try:
            os.mkdir("HotelTraffickingData/vertical")
        except FileExistsError:
            pass
        
        try:
            os.mkdir("HotelTraffickingData/horizaontal/val")
        except FileExistsError:
            pass

        try:
            os.mkdir("HotelTraffickingData/vertical/val")
        except FileExistsError:
            pass

        try:
            os.mkdir("HotelTraffickingData/horizaontal/val/" + str(hotel))
        except FileExistsError:
            pass

        try:
            os.mkdir("HotelTraffickingData/vertical/val/" + str(hotel))
        except FileExistsError:
            pass
        
        mask_path = "IM_Resize/masks/" + str(mask)
        image_path = "IM_Resize/images/" + str(hotel) + "/" + str(h_image)
        write_path_horizontal = "HotelTraffickingData/horizaontal/val/" + str(hotel) + str(f"/{count}.png")
        write_path_vertical = "HotelTraffickingData/vertical/val/" + str(hotel) + str(f"/{count}.png")
        
        loaded_mask = load_and_invert_mask(mask_path)
        masked_image = apply_mask(image_path, loaded_mask)

        im_array = blend_horizontal(masked_image)
        im = Image.fromarray(im_array)
        im.save(write_path_horizontal)

        im_array = blend_vertical(masked_image)
        im = Image.fromarray(im_array)
        im.save(write_path_vertical)

        with open(f"Support Files/Image Generation Summary/{hotel}_val.txt", "w") as fp:
            fp.write("\n".join(track_val))

        count += 1
    
    count = len(track_test) - 1
    for c_image in tqdm(test_images):
        track_test.append(c_image)
        c_image = c_image.split("/")
        hotel = c_image[0]
        h_image = c_image[1]
        mask = c_image[2]

        try:
            os.mkdir("HotelTraffickingData")
        except FileExistsError:
            pass

        try:
            os.mkdir("HotelTraffickingData/horizaontal")
        except FileExistsError:
            pass
        
        try:
            os.mkdir("HotelTraffickingData/vertical")
        except FileExistsError:
            pass
        
        try:
            os.mkdir("HotelTraffickingData/horizaontal/test")
        except FileExistsError:
            pass

        try:
            os.mkdir("HotelTraffickingData/vertical/test")
        except FileExistsError:
            pass

        try:
            os.mkdir("HotelTraffickingData/horizaontal/test/" + str(hotel))
        except FileExistsError:
            pass

        try:
            os.mkdir("HotelTraffickingData/vertical/test/" + str(hotel))
        except FileExistsError:
            pass
        
        mask_path = "IM_Resize/masks/" + str(mask)
        image_path = "IM_Resize/images/" + str(hotel) + "/" + str(h_image)
        write_path_horizontal = "HotelTraffickingData/horizaontal/test/" + str(hotel) + str(f"/{count}.png")
        write_path_vertical = "HotelTraffickingData/vertical/test/" + str(hotel) + str(f"/{count}.png")
        
        loaded_mask = load_and_invert_mask(mask_path)
        masked_image = apply_mask(image_path, loaded_mask)

        im_array = blend_horizontal(masked_image)
        im = Image.fromarray(im_array)
        im.save(write_path_horizontal)

        im_array = blend_vertical(masked_image)
        im = Image.fromarray(im_array)
        im.save(write_path_vertical)

        with open(f"Support Files/Image Generation Summary/{hotel}_test.txt", "w") as fp:
            fp.write("\n".join(track_test))

        count += 1