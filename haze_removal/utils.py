import matplotlib.pyplot as plt
import cv2


def resize_img(img, new_h=400):
    ratio = img.shape[1] / img.shape[0]
    new_w = int(new_h * ratio)
    return cv2.resize(img, (new_w, new_h))

def load_image(path, height=400):
    image = plt.imread(path) / 255
    image = resize_img(image, height)
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
