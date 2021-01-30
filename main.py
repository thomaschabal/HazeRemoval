# python main.py -p ./images/img.jpg

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from haze_removal import HazeRemover, load_image, show_imgs, PATCH_SIZE

parser = argparse.ArgumentParser(description="Haze removal function")
parser.add_argument("--path", "-p", type=str, help="Image path", default="./images/20210127_142511.jpg")
parser.add_argument("--patch_size", type=int, help="Patch size for dark channel extraction", default=PATCH_SIZE)
parser.add_argument("--height", type=int, help="Image height", default=500)
parser.add_argument("--save_folder", "-s", type=str, help="Folder to save haze-free image in", default="./results/")
args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder)
file_name = args.path.split('/')[-1]
name, extension = file_name.split('.')


print(f"Loading image from {args.path}")
image = load_image(args.path, args.height, show_image=False)

haze_remover = HazeRemover(image, patch_size=args.patch_size)
radiance, transmission, _ = haze_remover.remove_haze()

plt.imsave(args.save_folder + f"{name}_radiance.{extension}", radiance)
plt.imsave(args.save_folder + f"{name}_transmission.{extension}", transmission)
