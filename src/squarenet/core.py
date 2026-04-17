import numpy as np

""""
=============================================
=============================================
PSEUDO-CODE: convert unstructured point cloud (N, D) 
to a grid (N1, ..., ND, D) by iteratively sorting 
the d-est heuristic along the d-est axis

We can see the grid as D iterators on the points 
such that axis-d iterator maps  the point 
P(n ~ n1...nd...nD) to the "next" point
Pnext(n ~ n1...nd+1...nD). Let call it Pnext(n, d)

We can select D euclidian heuristics 
H(d): (x, y, z,...) -> value which we want to 
be increasing along the d-est axis of the grid
It turns out that H(0) = x, H(1) = y,.... is
already a pretty good heuristic.

So the goal is simply to ensure that all
heuristics are sorted along the grid, in the
sense that for all point P(n) and axis d,
H(d)(P(n)) <= H(d)(Pnext(n, d))

We can compute a grid disorder parameter which is
just the counts of all P, Pnext which breaks this 
inequality
=============================================
Sort_increasing is then pretty simple:
For learning step in (1, Max_iter = 100)
    For d in (1, D):
        sort heuristic d along axis d.
    Check disorder
    If disorder = 0, we are done !
=============================================
=============================================
"""

def sort_increasing(gridmap, heuristics, max_iter=100):
    """
    Args: 
        -gridmap (np.array of ints):
        an initial gridmap such that cloud_features[gridmap] 
        write any feature (N, *C) of the point-cloud (N, D) 
        on a grid (N1, ..., ND, *C)

        -max_iter (int):
        last step after which algorithm shall stop
        even if it hasn't converged yet
    Returns:
        -gridmap
        sorted gridmap such that cloud_features[gridmap]
        now write the feature on a spatially coherent grid
        -learningcurve (list of values):
        track the performance of the optimisation process.
        should converge to 0
    """
    g = gridmap.copy()
    learning_curve = []

    #loop[0]: index to heuristic 0
    #loop[d+1]: heuristic d to heuristic d+1
    #back_to_id: heuristic D-1 to index
    loop, back_to_id = loop_boost(heuristics)
    
    for _ in range(max_iter):
         # --- 1. Check for convergence ---
        disorder = 0
        for d, heuristic in enumerate(loop):
            g = heuristic[g]
            
            # Efficient disorder check: H(d)(P) > H(d)(Pnext)
            diff = np.diff(g, axis=d)
            disorder += np.sum(diff < 0)
        
        g = back_to_id[g]

        learning_curve.append(disorder)           
        if disorder == 0:
            break

        # --- 2. Sorting Phase ---
        for d, heuristic in enumerate(loop):
            g = heuristic[g]
            g.sort(axis = d)
            
        g = back_to_id[g]

    # last cleanup:
    gridmap = np.ascontiguousarray(g)
    return gridmap, learning_curve

# ============================================
# ============================================
# Heuristics
# ============================================
# ============================================
def carthesian_heuristics(points):
    return points

# ============================================
# ============================================
# Boosters
# ============================================
# ============================================
def integer_boost(heuristics):
    """
    Booster: Convert heuristics to integer 
    to boost sort_increasing function

    Args:
        - heuristics (np.ndarray) (N,D): heuristics computed on the point cloud

    Returns:
        - h_int (list of np.uint32 arrays): heuristics as integers
    """
    N, D = heuristics.shape
    h_int = []
    for d in range(D):
        order = np.argsort(heuristics[:, d])      
        ranks = np.empty(N, dtype=np.int32)
        ranks[order] = np.arange(N)
        h_int.append(ranks)
    return h_int

def loop_boost(heuristics):
    """
    Booster: make looping over heuristics a bit faster

    Args:
        - heuristics (np.ndarray) (N,D): heuristics computed on the point cloud
    Return:
        - loop (...): list of permutations such that 
        loop[d](h_int[d][n]) = h_int[d+1][n]
        - back_to_id (...): permutation such that 
        back_to_id(h_int[-1][n]) = n 
    """
    int_boost = integer_boost(heuristics)
    N = len(int_boost[0])
    identity = np.arange(N, dtype=np.int32)
    #start the loop
    h_int =  [identity] + int_boost
    #close the loop
    h_int_plus = int_boost + [identity]
    loop = []

    for h, hplus in zip(h_int, h_int_plus):
        sigma = np.zeros(N, dtype=np.int32)
        sigma[h] = hplus
        loop.append(np.ascontiguousarray(sigma))
        
    loop, back_to_id = loop[:-1], loop[-1]
    return loop, back_to_id