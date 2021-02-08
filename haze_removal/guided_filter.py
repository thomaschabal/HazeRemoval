import numpy as np
import cv2
from numba import jit
from .constants import EPS_GF

@jit
def extract_subpart2d(img, x, y, padding):
    h, w = img.shape
    x_start, x_end = max(0, x - padding), min(h, x + padding + 1)
    y_start, y_end = max(0, y - padding), min(w, y + padding + 1)
    return img[x_start:x_end, y_start:y_end]

@jit
def extract_subpart3d(img, x, y, padding):
    h, w = img.shape[:2]
    x_start, x_end = max(0, x - padding), min(h, x + padding + 1)
    y_start, y_end = max(0, y - padding), min(w, y + padding + 1)
    return img[x_start:x_end, y_start:y_end, :]

def compute_fast_guided_filter(guided_filter_function, input, guide_image, scale_factor=4, window_size=40, eps=EPS_GF):
    hs_input, ws_input = input.shape[0] // scale_factor, input.shape[1] // scale_factor
    hs_guide, ws_guide = guide_image.shape[0] // scale_factor, guide_image.shape[1] // scale_factor

    input_small = cv2.resize(input, (hs_input, ws_input), interpolation=cv2.INTER_AREA)
    guide_image_small = cv2.resize(guide_image, (hs_guide, ws_guide), interpolation=cv2.INTER_AREA)
    window_size = window_size // scale_factor

    mean_A_small, mean_B_small = guided_filter_function(input_small, guide_image_small, window_size, eps)

    w_guide, h_guide = guide_image.shape[:2]
    mean_A = cv2.resize(mean_A_small, (h_guide, w_guide), interpolation=cv2.INTER_LINEAR)
    mean_B = cv2.resize(mean_B_small, (h_guide, w_guide), interpolation=cv2.INTER_LINEAR)

    return mean_A, mean_B


# ========= USING GREY INPUT AS GUIDED IMAGE =================
@jit
def compute_guided_filter_grey(input, guide_image, window_size=40, eps=EPS_GF):
    A = np.zeros_like(guide_image)
    B = np.zeros_like(guide_image)
    padding = (window_size - 1) // 2

    h, w = input.shape[:2]
    for x in range(h):
        for y in range(w):
            patch_input = extract_subpart2d(input, x, y, padding)
            patch_guide = extract_subpart2d(guide_image, x, y, padding)

            mean_patch_input = np.mean(patch_input)
            mean_patch_guide = np.mean(patch_guide)
            mean_patch_input_guide = np.mean(patch_input * patch_guide)
            var_patch_guide = np.var(patch_guide)

            A[x, y] = 1 / (var_patch_guide + eps) * (mean_patch_input_guide - mean_patch_guide * mean_patch_input)
            B[x, y] = mean_patch_input - A[x, y] * mean_patch_guide

    mean_A = np.zeros_like(A)
    mean_B = np.zeros_like(B)

    for x in range(h):
        for y in range(w):
            mean_A[x, y] = np.mean(extract_subpart2d(A, x, y, padding))
            mean_B[x, y] = np.mean(extract_subpart2d(B, x, y, padding))

    return mean_A, mean_B


@jit
def guided_filter_grey(input, guide_image, window_size=40, eps=EPS_GF):
    mean_A, mean_B = compute_guided_filter_grey(input, guide_image, window_size, eps)
    return mean_A * guide_image + mean_B


def fast_guided_filter_grey(input, guide_image, scale_factor=4, window_size=40, eps=EPS_GF):
    mean_A, mean_B = compute_fast_guided_filter(compute_guided_filter_grey, input, guide_image, scale_factor, window_size, eps)
    return mean_A * guide_image + mean_B


# ========= USING COLORED INPUT AS GUIDED IMAGE =================
@jit
def compute_guided_filter_color(input, guide_image, window_size=30, eps=EPS_GF):
    A = np.zeros_like(guide_image)
    B = np.zeros_like(input)
    padding = (window_size - 1) // 2

    h, w = input.shape
    for x in range(h):
        for y in range(w):
            patch_input = extract_subpart2d(input, x, y, padding)
            patch_guide = extract_subpart3d(guide_image, x, y, padding)

            mean_patch_input = np.mean(patch_input)
            
            mean_patch_guide = np.zeros(3)
            mean_patch_input_guide = np.zeros(3)
            for i in range(3):
                mean_patch_guide[i] = np.mean(patch_guide[:,:,i])
                mean_patch_input_guide[i] = np.mean(patch_input * patch_guide[:,:,i])
            
            var_patch_guide = np.var(patch_guide)

            A[x, y, :] = np.dot(np.linalg.inv(var_patch_guide + eps * np.eye(3)), (mean_patch_input_guide - mean_patch_guide * mean_patch_input))
            B[x, y] = mean_patch_input - np.dot(A[x, y, :], mean_patch_guide)

    mean_A = np.zeros_like(A)
    mean_B = np.zeros_like(B)

    for x in range(h):
        for y in range(w):
            mean_A[x, y, :] = np.mean(extract_subpart3d(A, x, y, padding))
            mean_B[x, y] = np.mean(extract_subpart2d(B, x, y, padding))
    
    return mean_A, mean_B


@jit
def combine_meanA_meanB_guide(mean_A, guide_image, mean_B, input):
    output = np.zeros_like(input)
    h, w = input.shape
    for x in range(h):
        for y in range(w):
            output[x, y] = np.dot(mean_A[x,y,:], guide_image[x,y,:]) + mean_B[x,y]

    return output


@jit
def guided_filter_color(input, guide_image, window_size=30, eps=EPS_GF):
    mean_A, mean_B = compute_guided_filter_color(input, guide_image, window_size, eps)
    return combine_meanA_meanB_guide(mean_A, guide_image, mean_B, input)


def fast_guided_filter_color(input, guide_image, scale_factor=4, window_size=40, eps=EPS_GF):
    mean_A, mean_B = compute_fast_guided_filter(compute_guided_filter_color, input, guide_image, scale_factor, window_size, eps)
    return combine_meanA_meanB_guide(mean_A, guide_image, mean_B, input)
