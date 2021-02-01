# python main.py -p ./images/img.jpg --height 1000 --soft_matting False --guided_filtering True

import argparse
import matplotlib.pyplot as plt
from haze_removal import HazeRemover, load_image, show_imgs, PATCH_SIZE, create_save_folder_and_get_file_info, str2bool, get_save_extension

parser = argparse.ArgumentParser(description="Haze removal function")
parser.add_argument("--path", "-p", type=str, help="Image path", default="./images/20210127_142511.jpg")
parser.add_argument("--patch_size", type=int, help="Patch size for dark channel extraction", default=PATCH_SIZE)
parser.add_argument("--resize", type=int, help="Size of the largest side", default=1000)
parser.add_argument("--save_folder", "-s", type=str, help="Folder to save haze-free image in", default="./results/")
parser.add_argument("--soft_matting", "-m", type=str2bool, action="store_true", help="Boolean to use soft matting", default=False)
parser.add_argument("--guided_filtering", "-f", type=str2bool, action="store_true", help="Boolean to use guided filtering", default=True)
args = parser.parse_args()


name, file_extension, save_folder = create_save_folder_and_get_file_info(args.path, args.save_folder)

print(f"Loading image from {args.path}")
image = load_image(args.path, args.height, show_image=False)


haze_remover = HazeRemover(
    image,
    patch_size=args.patch_size,
    soft_matting=args.soft_matting,
    guided_image_filtering=args.guided_filtering
)
radiance, transmission, _ = haze_remover.remove_haze(.6)


extension = get_save_extension(args.soft_matting, args.guided_filtering, args.height, file_extension)
plt.imsave(save_folder + f"{name}_original_{extension}", image)
plt.imsave(save_folder + f"{name}_radiance_{extension}", radiance)
plt.imsave(save_folder + f"{name}_transmission_{extension}", transmission)
