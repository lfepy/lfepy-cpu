import numpy as np
from lfepy.Helper.gauss import gauss
from lfepy.Helper.dgauss import dgauss


def gauss_gradient(sigma):
    """
    Generate a set of 2-D Gaussian derivative kernels for gradient computation at multiple orientations.

    Args:
        sigma (float): The standard deviation of the Gaussian distribution.

    Returns:
        numpy.ndarray: A 3D array where each 2D slice represents a Gaussian derivative kernel at a specific orientation.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> sigma = 1.0
        >>> kernels = gauss_gradient(sigma)
        >>> fig, axes = plt.subplots(1, 8, figsize=(20, 5))
        >>> for i in range(8):
        ...     axes[i].imshow(kernels[:, :, i], cmap='gray')
        ...     axes[i].set_title(f'{i*45} degrees')
        ...     axes[i].axis('off')
        >>> plt.tight_layout()
        >>> plt.show()
    """
    epsilon = 1e-2
    halfsize = np.ceil(sigma * np.sqrt(-2 * np.log(np.sqrt(2 * np.pi) * sigma * epsilon)))
    size = int(2 * halfsize + 1)

    # Generate a 2-D Gaussian kernel along x direction
    hx = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            u = [i - halfsize - 1, j - halfsize - 1]
            hx[i, j] = gauss(u[0] - halfsize + 1, sigma) * dgauss(u[1], sigma)

    hx = hx / np.sqrt(np.sum(np.abs(hx) * np.abs(hx)))

    # Generate a 2-D Gaussian kernel along y direction
    D = np.zeros((hx.shape[0], hx.shape[1], 8))
    D[:, :, 0] = hx

    # Rotations using NumPy
    def rotate_image(image, angle):
        angle_rad = np.deg2rad(angle)
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        center = (np.array(image.shape) - 1) / 2
        coords = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]), indexing='ij')
        coords = np.stack(coords, axis=-1).astype(np.float32) - center
        new_coords = np.empty_like(coords)
        new_coords[..., 0] = coords[..., 0] * cos_angle - coords[..., 1] * sin_angle
        new_coords[..., 1] = coords[..., 0] * sin_angle + coords[..., 1] * cos_angle
        new_coords += center
        return np.clip(new_coords, 0, image.shape[0] - 1).astype(np.int32)

    for idx, angle in enumerate(range(45, 360, 45)):
        rotated_indices = rotate_image(hx, angle)
        rotated_image = np.zeros_like(hx)
        rotated_image[rotated_indices[..., 0], rotated_indices[..., 1]] = hx
        D[:, :, idx + 1] = rotated_image

    return D