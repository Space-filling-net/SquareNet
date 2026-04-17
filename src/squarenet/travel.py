import numpy as np

def travel(Cpoints: np.ndarray, k_neighbors: int = 5) -> np.ndarray:
    """
    Approximates the Traveling Salesman Problem (TSP) using a sparse K-Nearest Neighbors graph.
    
    This version is optimized for large datasets by limiting the search space to 
    the K closest neighbors for each point, significantly reducing memory 
    usage and computation time.

    Args:
        Cpoints (np.ndarray): 1D array of complex numbers representing 2D coordinates.
        k_neighbors (int): Number of nearest neighbors to consider for each point. 
                           Defaults to 10.

    Returns:
        np.ndarray: An array of indices representing the optimized path order.
    """
    import networkx as nx
    from scipy.spatial import KDTree
    n = len(Cpoints)
    if n <= 1:
        return np.arange(n)

    # Convert complex points to 2D real coordinates for KDTree
    coords = np.column_stack((Cpoints.real, Cpoints.imag))
    tree = KDTree(coords)

    # 1. Query the K-Nearest Neighbors
    # k+1 because the point itself is included in the results
    distances, indices = tree.query(coords, k=min(k_neighbors + 1, n))

    # 2. Build a sparse Graph
    G = nx.Graph()
    for i in range(n):
        for j_idx, dist in zip(indices[i], distances[i]):
            if i != j_idx:
                G.add_edge(i, j_idx, weight=dist)

    # 3. Solve TSP Approximation
    # This finds a Hamiltonian cycle in the sparse graph
    cycle = nx.approximation.traveling_salesman_problem(G, weight='weight')

    # 4. Convert cycle to path by breaking the longest edge
    path_nodes = np.array(cycle[:-1])  # Remove the duplicate last node
    
    # Calculate Euclidean distances between consecutive nodes in the cycle
    ordered_points = Cpoints[path_nodes]
    rolled_points = np.roll(ordered_points, -1)
    edge_weights = np.abs(ordered_points - rolled_points)
    
    # Find the index of the longest jump to break the loop
    max_idx = np.argmax(edge_weights)
    
    # Reorder (rotate) indices so the path starts after the longest edge
    sigma = np.roll(path_nodes, -(max_idx + 1))
    
    return sigma