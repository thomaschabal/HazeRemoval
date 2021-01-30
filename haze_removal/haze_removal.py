import numpy as np
import skimage.exposure as exposure
from tqdm import tqdm
from time import time
from scipy.ndimage import minimum_filter
from scipy.sparse import identity
from scipy.sparse.linalg import spsolve

from .constants import PATCH_SIZE, OMEGA, T0, LAMBDA, EPS, R
from .laplacian import compute_laplacian


class HazeRemover:
    def __init__(self, image, patch_size=PATCH_SIZE, omega=OMEGA, t0=T0, lambd=LAMBDA, eps=EPS, r=R, matting=True, print_intermediate=True):
        self.patch_size = patch_size
        self.omega = omega
        self.t0 = t0
        self.lambd = lambd
        self.print_intermediate = print_intermediate
        self.image = image
        self.matting = matting

        if matting:
            print("Computing matting laplacian...")
            start = time()
            self.laplacian = compute_laplacian(image, eps, r)
            if print_intermediate:
                print("Took {:2f}s to compute laplacian".format(time() - start))

    def extract_dark_channel(self, img):
        return np.min(minimum_filter(img, self.patch_size), axis=2)

    def compute_atmospheric_light(self):
        dark_channel = self.extract_dark_channel(self.image)
        n = int(1e-3 * np.prod(dark_channel.shape))
        brightest_dark_channel = np.argpartition(dark_channel.ravel(), -n)[-n:]

        maximum_intensity = 0
        self.atmospheric_light = None
        interest_zone = self.image.reshape((np.prod(dark_channel.shape), -1))[brightest_dark_channel]
        for pixel in interest_zone:
            intensity = np.sum(pixel)  #fixme: what is the definition?
            if intensity > maximum_intensity:
                maximum_intensity = intensity
                self.atmospheric_light = pixel

    def compute_transmission(self):
        dark_channel_normalized = self.extract_dark_channel(self.image / self.atmospheric_light[None, None])
        self.transmission = 1 - self.omega * dark_channel_normalized

    #TODO: ADD GUIDED FILTERING

    def soft_matting(self):
        start = time()
        A = self.laplacian + self.lambd * identity(self.laplacian.shape[0])
        b = self.lambd * self.transmission.ravel()
        self.transmission = spsolve(A, b).reshape(self.image.shape[:2])

        # self.transmission = np.clip(self.transmission, 0, 1)  #fixme

        if self.print_intermediate:
            print("Took {:2f}s to compute soft matte".format(time() - start))

    def compute_radiance(self):
        self.radiance = (self.image - self.atmospheric_light) / np.expand_dims(np.maximum(self.transmission, self.t0), -1) + self.atmospheric_light
        self.radiance = np.clip(self.radiance, 0, 1)
        # radiance = (radiance - np.min(radiance)) / (np.max(radiance) - np.min(radiance))


    def increase_exposure(self, value=1):
        self.radiance = exposure.adjust_gamma(self.radiance, gamma=value) # gain

    def remove_haze(self, correct_exposition=1):
        start = time()

        print("Computing atmospheric light...")
        self.compute_atmospheric_light()

        print("Computing transmission...")
        self.compute_transmission()

        if self.matting:
            print("Soft matting...")
            self.soft_matting()

        print("Computing radiance...")
        self.compute_radiance()

        print("Increasing exposure...")
        self.increase_exposure(correct_exposition)

        print("Took {:2f}s to perform haze removal".format(time() - start))

        return self.radiance, self.transmission, self.atmospheric_light
