import matplotlib.pyplot as plt
import cv2
from skimage.io import imread
from skimage.transform import resize

def load_image(path, maxwh=400, show_image=True):
    image = imread(path)
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
