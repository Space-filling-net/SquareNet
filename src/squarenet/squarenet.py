import numpy as np
from .core import sort_increasing, carthesian_heuristics
from .boards import checkerboard, checkerboard2D, checkerboard3D
from .views import localview, lazylocalview
from .utils import initpoint, dualgrid, dualgridflat
from warnings import warn

class SquareNet:
    """
    An iterative grid-straightening algorithm that untangles a D-dimensional mesh 
    by sorting nodes along each spatial axis to enforce a structured ordering.
    
    The grid is represented as an (I, J, ..., D) array, where D is the number of spatial dimensions.
    Each iteration attempts to minimize disorder along each dimension independently.
    """
    
    def __init__(self, IJ_=(100, 100), max_iter=100, warnings_=True):
        """
        Initialize the SquareNet.

        Args:
            IJ_ (tuple): Grid dimensions (Rows, Cols, ...). Supports any number of dimensions.
            max_iter (int): Maximum number of straightening iterations.
            warnings_ (bool): Flag to shut up all warnings if asked
        """
        self.IJ_ = tuple(IJ_)
        self.D = len(IJ_)  # Spatial dimensions (x, y, z,...)
        self.N = np.prod(IJ_)
        self.max_iter = max_iter
        self.learning_curve = []
        self.warnings_ = warnings_
        
        # Internal state
        self.points = None   # Points as given by the user
        self.heuristics = None # Heuristics is computed on points
        self.grid =  np.arange(
            self.N, dtype = np.int32
        ).reshape(IJ_) #grid to sort heuritics
        self.invert_grid = dualgrid(self.grid) #for invertibility
        self.invgridflat = dualgridflat(self.grid) #flat version
        self.packed = False #Flag for faster computations if possible

    def fit(self, points):
        """
        Fit the grid to a set of points in D dimensions.

        Args:
            points (np.ndarray or str): Array of shape (N, D) or a method name supported
                                        by .utils.initpoint.
        """
        if isinstance(points, str):
            points = initpoint(method=points, size=(self.N, self.D))
        
        N, D = points.shape
        assert N == self.N, f"Input points ({N}) must match grid size {self.N}"
        assert D == self.D, f"Input points dimension ({D}) must match D={self.D}"

        self.points = points
        self.heuristics = carthesian_heuristics(points)

        grid, learning_curve = sort_increasing(
            self.grid, self.heuristics, self.max_iter
        )

        last_iter = len(learning_curve) -1
        last_error = learning_curve[-1]

        if last_iter == 0:
            #just tacking advantage of situation 
            #to boost map and invertmap function
            self.autopack() 
        
        elif last_error == 0:
             print(f"succesfully sorted at iteration {last_iter}")

        else:
            if self.warnings_:
                warn(
                    "Disorder didn't converge to 0. "
                    "Check the learning curve and consider increasing the max_iter parameter.",
                    ConvergenceWarning,
                    stacklevel=2
                )
        
        # Save results
        self.grid = grid
        self.invert_grid = dualgrid(grid)
        self.invgridflat = dualgridflat(grid)
        self.learning_curve = learning_curve
    
    def map(self, features):
        """
        Gather: cloud data (N, *C) -> grid data (N1, ..., ND, *C)
        """
        if not self.packed:
            return features[self.grid]
        C = features.shape[1:]
        return features.reshape(*self.IJ_, *C)
    
    def invert_map(self, features):
        """
        Restores: grid data (N1, ..., ND, *C) -> cloud data (N, *C)
        """
        if not self.packed:
            return features.reshape(
                -1, *features.shape[self.D:]
                )[self.invgridflat]
        
        C = features.shape[self.D:]
        return features.reshape(-1, *C)
    
    def mapidx(self, index):
        """
        Convert: ONE cloud index -> ONE grid index
        """
        return self.invert_grid[index]
    
    def invert_mapidx(self, index):
        """
        Convert: ONE grid index-> ONE cloud index
        """
        return self.grid[tuple(index)]
               
    def checkerboard(self, scale = [2, 4, 8, 16], toplot = True, **kwargs):
        """
        alias for boards.checkerboard:

        return the net as a checkerboard at required scale = number of cells per axe
        make a nice plot if self.D = 2 or 3

        **kwargs are extra visualization arguments (color, point size...)
        """
        gpoints = np.ascontiguousarray(self.map(self.points))
        #nice plot if possible
        if toplot and (self.D == 2):
            checkerboard2D(gpoints, scale = scale, **kwargs)
        elif toplot and (self.D == 3):
            checkerboard3D(gpoints, scale = 8, **kwargs) 
        else:
            return checkerboard(gpoints, scale)
    
    def views(self, X, wr, map=True, invert_map=False, select_lazy=None, **kwargs):
        """
        Alias for views.localview

        Compute local neighborhood views of the input array `X`
        using rectangular windows of radius `wr`. The window size
        is given by ``ws = 2 * wr + 1`` (per dimension).

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (N, *C) or (*G, *C).
        wr : int or tuple (per dimension)
            Radius of the window.
        map : bool, default True
            If True, `X` is assumed to be in (N, *C) space.
            Otherwise, it is assumed to be in (*G, *C) space.
        invert_map : bool, default False
            If True, the output will stay in (*G, *C)  space 
            Otherwise, it will be converted back to (N, *C) 
        select_lazy : None or IndexLike, default None
            Subset of indices where the view is computed.
            Must be grid indexes
        **kwargs :
            Additional keyword arguments passed to `np.pad`
            (e.g., boundary conditions). See NumPy documentation.

        Returns
        -------
        Xview : np.ndarray or WindowCollector
            - If dense output:
                Array of shape (N, *C, *ws) or (*G, *C, *ws).
            - If `select_lazy` is used:
                A mapping (e.g. dict-like) from selected indices
                to local views of shape (*G[sel], *C, *ws).
        """
        lazy = (select_lazy is not None)

        Xmap = self.map(X) if map else X
        Xmap = np.ascontiguousarray(Xmap)

        args = (Xmap, wr, self.D)

        Xview = (
            localview(*args, **kwargs)
            if not lazy
            else lazylocalview(select_lazy, *args, **kwargs)
        )

        if invert_map:
            if lazy:
                Xview = {
                    self.invert_mapidx(key): value
                    for key, value in Xview.items()
                }
            else:
                if self.warnings_:
                    warn(
                        "Using invert_map may be slow and memory-intensive. "
                        "Consider working with gridded data or enabling select_lazy "
                        "if you only need a subset of the views.",
                        PerformanceWarning,
                        stacklevel=2
                    )
                Xview = self.invert_map(Xview)
         
        
        return Xview
    
    def autopack(self):
        print(
            "Successfully sorted at iteration 0...\n"
            "...Which means data are allready packed\n"
            "Performing an autopack to boost performance."
        )
        self.packed = True

        if self.warnings_:
            if not self.points.flags.c_contiguous:
                warn(
                    "Consider calling np.ascontiguousarray on your input arrays "
                    "(e.g., points and covariates) to improve performance.",
                    PerformanceWarning,
                    stacklevel=2
                )

class ConvergenceWarning(UserWarning):
    pass

class PerformanceWarning(UserWarning):
    pass