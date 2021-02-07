# From benchmarks folder:
# python evaluate_window_size_epsilon.py -p path/to/image -s ../results/window_size_epsilon/

import sys
sys.path.append("../")

import argparse
from haze_removal import load_image, create_save_folder_and_get_file_info
from haze_removal.benchmarks import evaluate_impact_of_window_size_and_epsilon


parser = argparse.ArgumentParser(description="Benchmark the parameters window_size and eps at once")
parser.add_argument("--path", "-p", type=str, help="Image path", default="./images/20210127_142511.jpg")
parser.add_argument("--resize", type=int, help="Size of the largest side", default=1000)
parser.add_argument("--save_folder", "-s", type=str, help="Folder to save haze-free image in", default="./results/window_size_epsilon/")
args = parser.parse_args()


_, _, save_folder = create_save_folder_and_get_file_info(args.path, args.save_folder)

print(f"Loading image from {args.path}")
image = load_image(args.path, args.resize, show_image=False)


WINDOW_SIZES = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 80, 100]
EPSILONS = [1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 1e-1, 1, 10, 100]
evaluate_impact_of_window_size_and_epsilon(WINDOW_SIZES, EPSILONS, image, save_folder)
