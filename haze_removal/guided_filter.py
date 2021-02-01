import numpy as np
from numba import jit


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


# ========= USING GREY INPUT AS GUIDED IMAGE =================
@jit
def guided_filter_grey_input(input, guide_image, window_size=40, eps=0.001):
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

    return mean_A * guide_image + mean_B


# ========= USING COLORED INPUT AS GUIDED IMAGE =================
@jit
def guided_filter_color_input(input, guide_image, window_size=40, eps=0.001):
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
    
    output = np.zeros_like(B)
    for x in range(h):
        for y in range(w):
            output[x, y] = np.dot(mean_A[x,y,:], guide_image[x,y,:]) + mean_B[x,y]
    
    return output
