import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

def get_grid_lines(shape_spatial, scales, linewidths):
    """
    Calcule les indices de lignes. 
    - 2D : Grille complète.
    - 3D : Uniquement sur les 3 faces "frontales" (max indices).
    """
    ndim = len(shape_spatial)
    base_steps = [max(1, s // 32) for s in shape_spatial]
    
    line_configs = []
    for scale, lw in zip(scales, linewidths):
        dls = [scale * bs for bs in base_steps]
        
        if ndim == 3:
            ni, nj, nk = shape_spatial
            # On ne cible que les indices max pour simuler les faces visibles
            imax, jmax, kmax = ni - 1, nj - 1, 0

            # Face latérale (J = jmax) et Face du haut (K = kmax) -> Lignes selon I
            for j in range(0, nj, dls[1]):
                line_configs.append({'indices': (slice(None), j, kmax), 'lw': lw})
            for k in range(0, nk, dls[2]):
                line_configs.append({'indices': (slice(None), jmax, k), 'lw': lw})

            # Face frontale (I = imax) et Face du haut (K = kmax) -> Lignes selon J
            for i in range(0, ni, dls[0]):
                line_configs.append({'indices': (i, slice(None), kmax), 'lw': lw})
            for k in range(0, nk, dls[2]):
                line_configs.append({'indices': (imax, slice(None), k), 'lw': lw})

            # Face frontale (I = imax) et Face latérale (J = jmax) -> Lignes selon K
            for i in range(0, ni, dls[0]):
                line_configs.append({'indices': (i, jmax, slice(None)), 'lw': lw})
            for j in range(0, nj, dls[1]):
                line_configs.append({'indices': (imax, j, slice(None)), 'lw': lw})

        else: # 2D : Inchangé, grille complète
            ni, nj = shape_spatial
            for j in range(0, nj, dls[1]):
                line_configs.append({'indices': (slice(None), j), 'lw': lw})
            for i in range(0, ni, dls[0]):
                line_configs.append({'indices': (i, slice(None)), 'lw': lw})
                
    return line_configs

def prepare_grid(grid, DS=1, axis = "auto"): 
    g = grid.copy()
    dims = np.array(g.shape[:-1]) 
    ndim = len(dims)

    if ndim >= 3:
        perm = np.argsort(dims)[::-1] if axis == "auto" else np.concatenate([axis, np.setdiff1d(np.arange(ndim), axis)])
        g = np.transpose(g, np.concatenate([perm, [-1]]))
        g = g[..., perm]
        slicer = [slice(None)] * ndim
        if ndim > 3:
            slicer[3:] = 0
        g = g[tuple(slicer)]
        g = g[..., :3]
        g = g[::DS, ::DS, ::DS]

        # Projection Isométrique
        alpha, theta = 0.5, np.pi / 4 
        P = np.array([
            [1, 0], [0, 1], 
            [alpha*np.cos(theta), alpha*np.sin(theta)]
        ])

        g = g @ P

         # Normalisation
        g_min, g_max = g.min(), g.max()
        g = (g - g_min) / (g_max - g_min + 1e-8)
        return g, "3D"
    else:
        g = g[::DS, ::DS, :2]
        g_min, g_max = g.min(), g.max()
        g = (g - g_min) / (g_max - g_min + 1e-8)
        return g, "2D"

def mesh(grid, ax, scales=[1, 2, 4, 8], linewidths=[0.2, 0.4, 0.7, 1.0]):
    ax.clear()
    line_configs = get_grid_lines(grid.shape[:-1], scales, linewidths)
    
    for config in line_configs:
        data = grid[config['indices']]
        ax.plot(data[..., 0], data[..., 1], color="black", lw=config['lw'], alpha=0.8)

    ax.set_aspect("equal")
    ax.axis("off")

def plot_mesh(grid_data):
    fig, ax = plt.subplots(figsize=(10, 10))
    prepared, _ = prepare_grid(grid_data)
    mesh(prepared, ax)
    plt.show()
    plt.close(fig)
    return

def animate_mesh(grid, save_path="animation_mesh.gif", method="fast", axis = "auto"):
    # Paramétrage
    if method == "fast":
        DS, frames, figsize = 2, 25, (6, 6)
        scales, lws = [4], [0.8]
    else:
        DS, frames, figsize = 1, 60, (10, 10)
        scales, lws = [2, 4, 8], [0.3, 0.6, 1.0]
    
    # Préparation
    grid_data, mode = prepare_grid(grid, DS=DS, axis = axis)
    shape_spatial = grid_data.shape[:-1]
    
    # Identité (même logique que ton code original)
    coords = [np.linspace(0, 1, s) for s in shape_spatial]
    identity_raw = np.stack(np.meshgrid(*coords, indexing='ij'), axis=-1)
    
    if mode == "3D":
        alpha, theta = 0.5, np.pi / 4 
        P = np.array([[1, 0], [0, 1], [alpha*np.cos(theta), alpha*np.sin(theta)]])
        identity = identity_raw @ P
    else:
        identity = identity_raw
    imin, imax =  identity.min(), identity.max()
    identity = (identity - imin)/ (imax - imin + 1e-8)

    # Setup Figure
    fig, ax = plt.subplots(figsize=figsize)
    x_min, x_max = 0, 1
    y_min, y_max = 0, 1
    margin = 0.2
    
    # Setup Figure
    ax.set_aspect("equal")
    
    # Appliquer les limites pour centrer
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    
    ax.axis("off")
    
    # Initialisation des lignes via le moteur factorisé
    line_configs = get_grid_lines(shape_spatial, scales, lws)
    plot_objects = []
    
    for conf in line_configs:
        ln, = ax.plot([], [], color="black", lw=conf['lw'], animated=True, alpha=0.7)
        plot_objects.append({'obj': ln, 'indices': conf['indices']})

    def update(frame):
        # Ping-pong timing
        t_raw = (frame / (frames - 1)) * 2
        t = t_raw if t_raw <= 1 else 2 - t_raw
        
        current = (1 - t) * grid_data + t * identity

        for item in plot_objects:
            data = current[item['indices']]
            item['obj'].set_data(data[..., 0], data[..., 1])

        return [l['obj'] for l in plot_objects]

    anim = FuncAnimation(fig, update, frames=frames, interval=50, blit=True)
    
    if save_path:
        anim.save(save_path, fps=16)
    plt.close(fig)
    return 