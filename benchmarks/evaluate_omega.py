# From benchmarks folder:
# python evaluate_omega.py -p path/to/image -s ../results/omega/

import sys
sys.path.append("../")

import argparse
from haze_removal import load_image, create_save_folder_and_get_file_info
from haze_removal.benchmarks import evaluate_impact_of_omega


parser = argparse.ArgumentParser(description="Benchmark the parameter omega")
parser.add_argument("--path", "-p", type=str, help="Image path", default="./images/20210127_142511.jpg")
parser.add_argument("--resize", type=int, help="Size of the largest side", default=1000)
parser.add_argument("--save_folder", "-s", type=str, help="Folder to save haze-free image in", default="./results/omega/")
args = parser.parse_args()


_, _, save_folder = create_save_folder_and_get_file_info(args.path, args.save_folder)

print(f"Loading image from {args.path}")
image = load_image(args.path, args.resize, show_image=False)


OMEGAS = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.98, 1]
evaluate_impact_of_omega(OMEGAS, image, save_folder)