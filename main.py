# python main.py -p ./images/img.jpg --resize 800

import argparse
import matplotlib.pyplot as plt
from haze_removal import HazeRemover, load_image, create_save_folder_and_get_file_info, get_save_extension, LAMBDA, T0, OMEGA, OPAQUE, GAMMA, PATCH_SIZE

parser = argparse.ArgumentParser(description="Haze removal function")
parser.add_argument("--path", "-p", type=str, help="Image path", default=None)
parser.add_argument("--patch_size", type=int, help="Patch size for dark channel extraction", default=PATCH_SIZE)
parser.add_argument("--lambd", type=float, default=LAMBDA)
parser.add_argument("--t0", type=float, default=T0)
parser.add_argument("--omega", type=float, default=OMEGA)
parser.add_argument("--opaque", type=float, default=OPAQUE)
parser.add_argument("--gamma", type=float, default=GAMMA)
parser.add_argument("--resize", type=int, help="Size of the largest side", default=1400)
parser.add_argument("--save_folder", "-s", type=str, help="Folder to save haze-free image in", default="./results/")
parser.add_argument("--soft_matting", "-m", action='store_true', help="Boolean to use soft matting")
parser.add_argument("--guided_filtering", "-f", action='store_true', help="Boolean to use guided filtering")
args = parser.parse_args()


name, file_extension, save_folder = create_save_folder_and_get_file_info(args.path, args.save_folder)

image = load_image(args.path, args.resize, show_image=False)


haze_remover = HazeRemover(
    image,
    patch_size=args.patch_size,
    use_soft_matting=args.soft_matting,
    guided_image_filtering=args.guided_filtering,
    lambd=args.lambd,
    t0=args.t0,
    omega=args.omega,
    opaque=args.opaque,
)
radiance, transmission, _ = haze_remover.remove_haze(args.gamma)


extension = get_save_extension(args.soft_matting, args.guided_filtering, args.resize, file_extension)
plt.imsave(save_folder + f"{name}_original.{file_extension}", image)
plt.imsave(save_folder + f"{name}_radiance_{extension}", radiance)
plt.imsave(save_folder + f"{name}_transmission_{extension}", transmission, cmap='gray')
