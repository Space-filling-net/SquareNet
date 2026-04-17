import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def checkerboard(grid, scale):
    """Split a grid into two sets of blocks following a checkerboard pattern."""
    ni, nj = grid.shape[:2]
    d = grid.shape[-1]
    hi, hj = ni // scale, nj // scale

    # Crop to ensure clean divisibility
    grid = grid[:hi * scale, :hj * scale]

    # Reshape and transpose into a block structure
    blocks = grid.reshape(scale, hi, scale, hj, d)
    blocks = blocks.transpose(0, 2, 1, 3, 4)

    # Flatten into blocks for easy masking
    flat_blocks = blocks.reshape(scale, scale, hi * hj, d)

    # Generate checkerboard mask
    ii, jj = np.indices((scale, scale))
    mask = (ii % 2) == (jj % 2)

    return flat_blocks[mask], flat_blocks[~mask]

def checkerboard2D(grid, scale=[2, 4, 8, 16], ax=None, colors=("lightgrey", "blue"), s=1, alpha=0.7):
    """Visualize different checkerboard scales on a 2D/3D point grid."""
    # 1. Handle multiple scales
    if isinstance(scale, list):
        ns = int(np.sqrt(len(scale)))
        fig, axes = plt.subplots(ns, ns, figsize=(10, 10))
        for a, sc in zip(axes.flat, scale):
            checkerboard2D(grid, scale=sc, ax=a, colors=colors, s=s, alpha=alpha)
        return axes

    # 2. Base case: single scale
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    
    # Use the checkerboard function to get the two sets of points
    set1, set2 = checkerboard(grid, scale)

    # Plot both sets (set1 are all 'blue' blocks, set2 are all 'lightgrey' blocks)
    for points, color in zip([set1, set2], colors):
        # points shape: (num_blocks, points_per_block, dims) -> flat for scatter
        pts_flat = points.reshape(-1, grid.shape[-1])
        ax.scatter(*(pts_flat[:, i] for i in range(pts_flat.shape[-1])), 
                   c=color, s=s, alpha=alpha, edgecolors='none')

    ax.set_title(f"Scale: {scale}")
    ax.axis("off")
    return ax

def checkerboard3D(grid_3d, scale=8, ax = None, colors=("lightgrey", "blue"), s=3, alpha=1):
    """Visualize different checkerboard scales on the surface of a 3D point grid."""
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

    faces = [
        grid_3d[-1, :, :], #grid_3d[0, :, :],
        grid_3d[:, 0, :], #grid_3d[:, -1, :],
        grid_3d[:, :, -1], #grid_3d[:, :, 0]
    ]

    for face in faces:
        checkerboard2D(face, scale=scale, ax=ax, colors = colors, s=s, alpha=alpha)

    ax.set_box_aspect([1, 1, 1])
    plt.show()