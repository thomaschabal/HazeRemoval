import os
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize


def load_image(path, maxwh=400, show_image=True):
    print(f"Loading image from {path}")
    image = imread(path) / 255
    h, w = image.shape[:2]
    if max(h, w) > maxwh:
        if h > w:
            image = resize(image, (maxwh, int(w*maxwh/h)))
        else:
            image = resize(image, (int(h*maxwh/w), maxwh))
    if show_image:
        plt.imshow(image)
        plt.axis('off')
        plt.title('Initial image')
        plt.show()
    return image


def show_imgs(imgs, figsize=(20,10)):
    plt.figure(figsize=figsize)
    for ind, img in enumerate(imgs):
        plt.subplot(len(imgs)//3 + 1, len(imgs) % 3 + 1, ind+1)
        plt.imshow(img)
        plt.axis('off')
    plt.show()


def create_save_folder_and_get_file_info(file_path, save_folder):
    if save_folder[-1] != '/':
        save_folder += '/'

    file_name = file_path.split('/')[-1]
    name, extension = file_name.split('.')
    save_folder += name + '/'

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    return name, extension, save_folder


def get_save_extension(soft_matting, guided_filtering, height, file_extension):
    if soft_matting:
        method_type = "soft_matting"
    elif guided_filtering:
        method_type = "guided_filtering"
    else:
        method_type = "basic"
    extension = f"{method_type}_{height}px.{file_extension}"

    return extension
