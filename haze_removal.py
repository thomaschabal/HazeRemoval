import numpy as np
import cv2
from tqdm import tqdm
from time import time

from constants import PATCH_SIZE, OMEGA, T0


class HazeRemover:
    def __init__(self, image, patch_size=PATCH_SIZE, omega=OMEGA, t0=T0, print_intermediate=True):
        self.patch_size = patch_size
        self.omega = omega
        self.t0 = t0
        self.print_intermediate = print_intermediate
        self.image = image

    def extract_dark_channel(self, A=np.array([1,1,1])):
        start = time()
        min_per_patch_and_channel = np.zeros_like(self.image)

        patch_side_size = self.patch_size // 2
        h, w = self.image.shape[:2]
        for x in tqdm(range(h)):
            for y in range(w):
                min_x, max_x = max(0, x - patch_side_size), min(x + patch_side_size, h)
                min_y, max_y = max(0, y - patch_side_size), min(y + patch_side_size, w)
                for c in range(self.image.shape[2]):
                    min_per_patch_and_channel[x, y, c] = np.min(self.image[min_x:max_x, min_y:max_y, c]) / A[c]

        if self.print_intermediate:
            print("Took {:2f}s to compute dark channel".format(time() - start))

        self.dark_channel = np.min(min_per_patch_and_channel, axis=2)

    
    def compute_atmospheric_light(self):
        sorted_darkness = np.sort(self.dark_channel).flatten()
        values_to_keep = int(0.1 / 100 * sorted_darkness.shape[0])
        top_brightest_dark_channel = sorted_darkness[:values_to_keep]
        
        keep_dark_channel = np.where(self.dark_channel >= np.min(top_brightest_dark_channel), 1, 0)

        maximum_intensity = 0
        atmospheric_light = None
        h, w = self.image.shape[:2]
        for x in tqdm(range(h)):
            for y in range(w):
                if keep_dark_channel[x, y]:
                    pixel = self.image[x, y, :]
                    intensity = np.linalg.norm(pixel)
                    if intensity >= maximum_intensity:
                        maximum_intensity = intensity
                        atmospheric_light = pixel

        self.atmospheric_light = atmospheric_light


    def compute_transmission(self):
        self.extract_dark_channel(self.atmospheric_light)
        self.transmission = 1 - self.omega * self.dark_channel

    ## ADD SOFT MATTING AND GUIDED FILTERING
    

    def compute_radiance(self):
        start = time()

        transmission_bounded = np.where(self.transmission >= self.t0, self.transmission, self.t0)
        transmission_bounded = cv2.merge((transmission_bounded, transmission_bounded, transmission_bounded))

        self.radiance = (self.image - self.atmospheric_light) / transmission_bounded + self.atmospheric_light

        if self.print_intermediate:
            print("Took {:2f}s to compute radiance".format(time() - start))


    def remove_haze(self):
        start = time()

        print("Extracting dark channel...")
        self.extract_dark_channel()

        print("Computing atmospheric light...")
        self.compute_atmospheric_light()

        print("Computing transmission...")
        self.compute_transmission()

        print("Computing radiance...")
        self.compute_radiance()

        print("Took {:2f}s to perform haze removal".format(time() - start))
        
        return self.radiance, self.transmission, self.atmospheric_light
