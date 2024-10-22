import numpy as np

def compute_distances(data, metric='euclidean'):
    if (metric == 'euclidean'):
        return np.linalg.norm(data[:, np.newaxis] - data, axis=2)
    elif (metric == 'cosine'):
        normalized_data = data / np.linalg.norm(data, axis=1, keepdims=True)
        return 1 - np.dot(normalized_data, normalized_data.T)
    elif (metric == 'dot_product'):
        return -np.dot(data, data.T)
    else:
        raise ValueError("Unsupported metric: choose 'euclidean', 'cosine', or 'dot_product'")

def greedy_coreset_sampling(data, coreset_size, initial_indices, metric='euclidean'):
    """
    Perform greedy coreset sampling on the given data with specified initial points and distance metric.
    
    Parameters:
    data (np.ndarray): The data points, shape (N, D) where N is the number of points and D is the dimension.
    coreset_size (int): The number of points to select for the coreset.
    initial_indices (list or np.ndarray): Indices of the initial points to include in the coreset.
    metric (str): The distance metric to use ('euclidean', 'cosine', or 'dot_product').
    
    Returns:
    np.ndarray: Indices of the selected points.
    """
    num_points = data.shape[0]
    coreset_indices = list(initial_indices)
    
    # Compute pairwise distances or similarities
    distances = compute_distances(data, metric)
    
    # Initialize minimum distances to infinity
    min_distances = np.full(num_points, np.inf)
    
    # Update minimum distances based on initial points
    for idx in initial_indices:
        min_distances = np.minimum(min_distances, distances[idx])
    
    for _ in range(len(initial_indices), coreset_size):
        # Select the point with the maximum minimum distance to the coreset
        next_index = np.argmax(min_distances)
        coreset_indices.append(next_index)
        
        # Update minimum distances
        min_distances = np.minimum(min_distances, distances[next_index])
    
    return np.array(coreset_indices)


if __name__ == '__main__':
    # Example usage
    data = np.random.randn(100, 128)  # 100 points in 128-dimensional space
    coreset_size = 20
    initial_indices = np.random.choice(100, 10, replace=False)  # Randomly select 10 initial points
    metric = 'cosine'  # Choose 'euclidean', 'cosine', or 'dot_product'
    coreset_indices = greedy_coreset_sampling(data, coreset_size, initial_indices, metric)
    print("Selected coreset indices:", coreset_indices)