import numpy as np
import skimage.exposure as exposure
from warnings import warn
from time import time
from scipy.ndimage import minimum_filter
from scipy.sparse import identity, diags
from scipy.sparse.linalg import cg, aslinearoperator

from .constants import PATCH_SIZE, OMEGA, T0, LAMBDA, EPS, R, OPAQUE
from .laplacian import compute_laplacian
from .guided_filter import guided_filter_grey, guided_filter_color, fast_guided_filter_grey, fast_guided_filter_color


class HazeRemover:
    def __init__(self, image, patch_size=PATCH_SIZE, omega=OMEGA, t0=T0, lambd=LAMBDA, eps=EPS, r=R, opaque=OPAQUE,  window_size=None, use_soft_matting=True, guided_image_filtering=False, fast_guide_filter=True, print_intermediate=True):
        self.patch_size = patch_size
        self.omega = omega
        self.t0 = t0
        self.lambd = lambd
        self.opaque = opaque
        self.eps = eps
        self.r = r
        self.window_size = window_size
        self.print_intermediate = print_intermediate
        self.image = image
        self.use_soft_matting = use_soft_matting
        self.guided_image_filtering = guided_image_filtering
        self.fast_guide_filter = fast_guide_filter

    def extract_dark_channel(self, img):
        return np.min(minimum_filter(img, self.patch_size), axis=2)

    def compute_atmospheric_light(self):
        dark_channel = self.extract_dark_channel(self.image)
        n = int(self.opaque * np.prod(dark_channel.shape))
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

    def soft_matting(self):
        print("Computing matting laplacian...")
        start = time()
        laplacian = compute_laplacian(self.image, self.eps, self.r)
        if self.print_intermediate:
            print("Took {:2f}s to compute laplacian".format(time() - start))

        print("Soft matting...")
        start = time()
        A = laplacian + self.lambd * identity(laplacian.shape[0])
        b = self.lambd * self.transmission.ravel()
        M = aslinearoperator(diags(1.0 / A.diagonal()))
        tmp, s = cg(A, b, M=M, maxiter=1000)
        if s == 0:
            self.transmission = tmp.reshape(self.image.shape[:2])
        else:
            warn("Failed to compute soft matte")

        # self.transmission = np.clip(self.transmission, 0, 1)  #fixme

        if self.print_intermediate:
            print("Took {:2f}s to compute soft matte".format(time() - start))

    def guided_filtering(self):
        start = time()
        window_size = int(10 * (np.sqrt(max(self.image.shape[:2])) // 10)) if self.window_size is None else self.window_size

        # ========= USING GREY INPUT AS GUIDED IMAGE =================
        refinement_method = fast_guided_filter_grey if self.fast_guide_filter else guided_filter_grey

        # ========= USING COLORED INPUT AS GUIDED IMAGE =================
        # refinement_method = fast_guided_filter_color if self.fast_guide_filter else guided_filter_color

        self.transmission = refinement_method(self.transmission, self.image[:,:,0], window_size=window_size)
        if self.print_intermediate:
            print("Took {:2f}s to perform guided filtering".format(time() - start))

    def compute_radiance(self):
        self.radiance = (self.image - self.atmospheric_light) / np.expand_dims(np.maximum(self.transmission, self.t0), -1) + self.atmospheric_light
        if (self.radiance < 0).any() or (self.radiance > 1).any():
            warn("Clipping radiance")
            self.radiance = np.clip(self.radiance, 0, 1)


    def increase_exposure(self, value=1):
        self.radiance = exposure.adjust_gamma(self.radiance, gamma=value) # gain

    def remove_haze(self, correct_exposition=1):
        start = time()

        print("Computing atmospheric light...")
        self.compute_atmospheric_light()

        print("Computing transmission...")
        self.compute_transmission()

        if self.use_soft_matting:
            self.soft_matting()
        elif self.guided_image_filtering:
            print("Guided filtering...")
            self.guided_filtering()

        print("Computing radiance...")
        self.compute_radiance()

        print("Increasing exposure...")
        self.increase_exposure(correct_exposition)

        print("Took {:2f}s to perform haze removal".format(time() - start))

        return self.radiance, self.transmission, self.atmospheric_light
