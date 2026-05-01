import numpy as np
import time

def index_identity(shape):
     #index_identity[i,j,k] = [i, j, k]
     return np.moveaxis(np.indices(shape), 0, -1)

def dualgrid(grid):
        #dual_grid[grid] = identity
        IJ_ = grid.shape
        N, D = np.prod(IJ_), len(IJ_)
        dual_grid =  np.zeros((N, D), dtype = np.int32)
        dual_grid[grid] = index_identity(IJ_)

        #important, for efficient storage
        dual_grid = np.ascontiguousarray(dual_grid)
        return dual_grid

def dualgridflat(grid):
    gr= grid.flatten()
    N = len(gr)
    dgf = np.zeros(N, dtype = np.int32)
    identity = np.arange(N, dtype = np.int32)
    dgf[gr] = identity
    dgf = np.ascontiguousarray(dgf)
    return dgf

class Potential:
    """Simple O(N²) potential wrapper with safety estimates."""

    def __init__(self, potential_func, timeout=10, verbose=True, batchfirst = False, **params):
        self.f = potential_func
        self.timeout = timeout
        self.verbose = verbose
        self.batchfirst = batchfirst
        self.params = params
    def _estimate(self, x, y):
        """
        Estimate complexity for naive computation of x, y interaction
        """
        first = 1 if self.batchfirst  else 0
        nb = x.shape[0] if self.batchfirst else 1
        nx, ny = x.shape[first], y.shape[first]
        d = x.shape[-1]

        mem_gb = nb*nx * ny * d * 8 / 1e9
        est_time = nb * nx * ny * d * 1e-9  # rough O(N²)

        return mem_gb, est_time

    def _run(self, x, y, func):
        """
        Naively compute x,y interaction if possible, return complexity if failed
        """
        start = time.time()

        mem_gb, est_time = self._estimate(x, y)

        try:
            out = func()

            dt = time.time() - start
            if dt > self.timeout:
                raise TimeoutError()

            if self.verbose:
                print(f"[OK] computed in {dt:.3f}s")

            return out

        except Exception:
            print(
                "[Crash] failed due to O(N²) complexity | "
                f"estimated_memory={mem_gb:.2f} GB | "
                f"estimated_runtime={est_time:.2f} s"
            )
            return None
        
    def interaction(self, x, y):
        """
        Naive interaction matrix
        """
        if self.batchfirst:
            xx, yy = x[:, :, None, :], y[:, None, :, :]
        else:
            xx, yy = x[:, None, :], y[None, :, :]
        return self._run(
            x, y,
            lambda: self.f(xx - yy, **self.params),
        )
    
    def energy(self, x):
        """
        Naive total energy
        """
        if self.batchfirst:
            xx, yy = x[:, :, None, :], x[:, None, :, :]
        else:
            xx, yy = x[:, None, :], x[None, :, :]
        return self._run(
            x, x,
            lambda: np.sum(
                self.f(xx - yy, **self.params)
            ) / 2
        )

    def __call__(self, x):
        return self.energy(x)
    

def breakpoint():
    raise RuntimeError("STOP checkpoint.\n Everything allright...")