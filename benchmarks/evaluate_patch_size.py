# From benchmarks folder:
# python evaluate_patch_size.py -p path/to/image -s ../results/patch_size/

import sys
sys.path.append("../")

import argparse
from haze_removal import load_image, create_save_folder_and_get_file_info
from haze_removal.benchmarks import evaluate_impact_of_patch_size


parser = argparse.ArgumentParser(description="Benchmark the parameter patch_size")
parser.add_argument("--path", "-p", type=str, help="Image path", default=None)
parser.add_argument("--resize", type=int, help="Size of the largest side", default=1400)
parser.add_argument("--save_folder", "-s", type=str, help="Folder to save haze-free image in", default="./results/patch_size/")
args = parser.parse_args()


_, _, save_folder = create_save_folder_and_get_file_info(args.path, args.save_folder)

image = load_image(args.path, args.resize, show_image=False)


PATCH_SIZES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 80, 100]
evaluate_impact_of_patch_size(PATCH_SIZES, image, save_folder)
