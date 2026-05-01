import numpy as np

def cylindric_rotation(geometry, n_iter=30):
    """Sample a random rotation which respects aproximate
    rotation  symetries of a given geometry (i.e. cylinders)
    and apply it to the eigenvectors

    Args:
        geometry = (eigvals, eigvects)
    
    Returns:
        eigvects: new (rotated) eigenvectors
    """
    eigvals, eigvects = geometry
    eigvals = eigvals.copy()
    eigvects = eigvects.copy()
    d = len(eigvals)

    #trackers for permutations and matches
    sigma = np.arange(d)
    matches = n_iter * np.eye(d, dtype = int)


    # safe symetric division
    denom = np.maximum(eigvals, 1e-12)[:, None]
    num = np.maximum(eigvals, 1e-12)[None, :]
    ratio = num/denom
    min_ratio = np.minimum(ratio, ratio.T)
    max_theta = np.arctan(min_ratio)

    #rotation plan
    rotation_plan = max_theta * np.random.rand(d, d)


    def apply(sigma, arrays):
        for i in range(len(arrays)):
            arrays[i][:] = arrays[i][sigma]

    def rotate(x, y, theta):
        cos_t = np.cos(theta[:, None])
        sin_t = np.sin(theta[:, None])
        x_new = cos_t * x - sin_t * y
        y_new = sin_t * x + cos_t * y
        return x_new, y_new

    while np.min(matches) <= n_iter-1:
        permutation = np.random.permutation(d)
        apply(permutation, [sigma, eigvals, eigvects])
        theta = rotation_plan[sigma[:-1:2], sigma[1::2]]/n_iter
        x = eigvects[:-1:2]
        y = eigvects[1::2]

        x_new, y_new = rotate(x, y, theta)

        eigvects[:-1:2] = x_new
        eigvects[1::2] = y_new

        matches[sigma[:-1:2], sigma[1::2]] +=1

    #back to the original sequence
    eigvects = eigvects[sigma.argsort()]
    return eigvects