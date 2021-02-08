# From benchmarks folder:
# python launch_all_evaluations.py -f ../image -s ../results/

import sys
sys.path.append("../")

import argparse
import subprocess
from tqdm import tqdm


EVALUATIONS = {
    "evaluate_epsilon": "epsilon",
    "evaluate_matting": "matting",
    "evaluate_omega": "omega",
    "evaluate_patch_size": "patch_size",
    "evaluate_t0": "t0",
    "evaluate_window_size_epsilon": "window_size_epsilon",
    "evaluate_window_size": "window_size",
}


parser = argparse.ArgumentParser(description="Run all the evaluation scripts")
parser.add_argument("--folder", "-f", type=str, help="Images folder")
parser.add_argument("--resize", type=int, help="Size of the largest side", default=1400)
parser.add_argument("--save_folder", "-s", type=str, help="Folder to save haze-free image in", default="./results/")
args = parser.parse_args()


files = subprocess.check_output(["ls", args.folder])

for line in tqdm(files.splitlines()):
    file_name = line.decode('ascii')
    for script, subfolder in EVALUATIONS.items():
        subprocess.check_output([
            "python", f"{script}.py",
            "-p", "../images/" + file_name,
            "-s", f"../results/{subfolder}/",
            "--resize", str(args.resize),
        ])
