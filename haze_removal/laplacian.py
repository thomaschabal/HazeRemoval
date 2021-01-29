import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix


#@jit
def compute_laplacian(image, eps, r):
    h, w = image.shape[:2]
    n = h * w

    area = (2 * r + 1) ** 2

    indices = np.arange(n).reshape(h, w)

    values = np.zeros((n, area ** 2))
    i_inds = np.zeros((n, area ** 2), dtype=np.int32)
    j_inds = np.zeros((n, area ** 2), dtype=np.int32)

    # gray = (image[:, :, 0] + image[:, :, 1] + image[:, :, 2]) / 3.0
    # v = np.std(gray)

    for y in tqdm(range(r, h - r)):
        for x in range(r, w - r):
            i = x + y * w

            X = np.ones((area, 3 + 1))

            k = 0
            for y2 in range(y - r, y + r + 1):
                for x2 in range(x - r, x + r + 1):
                    for c in range(3):
                        X[k, c] = image[y2, x2, c]
                    k += 1

            window_indices = indices[y - r : y + r + 1, x - r : x + r + 1].flatten()

            K = np.dot(X, X.T)

            f = np.linalg.solve(K + eps * np.eye(area), K)

            tmp2 = np.eye(f.shape[0]) - f
            tmp3 = tmp2.dot(tmp2.T)

            for k in range(area):
                i_inds[i, k::area] = window_indices
                j_inds[i, k * area : k * area + area] = window_indices
            values[i] = tmp3.ravel()

    return csr_matrix((values.ravel(), (i_inds.ravel(), j_inds.ravel())), shape=(n, n))
