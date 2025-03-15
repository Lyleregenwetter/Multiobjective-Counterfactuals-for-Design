import numpy as np

from decode_mcd_private.stats_methods import mixed_gower

def compute_dynamic_batch_size(len_idx1, sample_dim, memory_target_gb=1):
    """
    Computes a safe batch size that keeps `cdist` memory usage within limits.
    Adjusts for intermediate computations across feature dimensions `d`.
    """
    bytes_per_float = 8  # float64
    memory_target_bytes = memory_target_gb * (1024 ** 3)  # Convert GB to Bytes

    # Compute batch size considering intermediate memory for `d` dimensions
    batch_size = memory_target_bytes // (2 * len_idx1 * sample_dim * bytes_per_float)

    # Ensure batch size is at most len(idx2) and at least 1
    return max(1, batch_size)

def min_weighted_distance(x_df, y, idx1, idx2, problem, memory_target_gb=4):
    """
    Computes the minimum weighted mixed Gower distances from each sample in idx1 to any sample in idx2.
    Uses batch processing to avoid excessive memory use.
    """
    min_dists = np.full(len(idx1), np.inf, dtype=np.float64)

    # Compute batch size dynamically based on data dimension
    sample_dim = x_df.shape[1]  # Number of features
    batch_size = compute_dynamic_batch_size(len(idx1), sample_dim, memory_target_gb)

    for j in range(0, len(idx2), batch_size):
        batch_idx2 = idx2[j:j+batch_size]

        # Compute mixed Gower distances for the batch
        dists = mixed_gower(x_df.iloc[idx1], x_df.iloc[batch_idx2], 
                            problem._ranges.values, problem._build_feature_types())

        # Apply weighting
        weighted_dists = np.einsum('ij,i,j->ij', dists, y[idx1], y[batch_idx2])

        # Update minimum distances
        min_dists = np.minimum(min_dists, weighted_dists.min(axis=1))

    return min_dists

def MMDP_sample(x_df, y, num_samples, problem, MMDP_steps=1, memory_target_gb=4):
    """
    Selects a diverse subset of samples using a greedy max-min strategy,
    followed by a Drop-Add optimization process, optimized for mixed data types.

    Parameters:
        x_df (pd.DataFrame): Dataset with mixed data types.
        y (np.ndarray): Weighting factors for each sample.
        num_samples (int): Number of samples to select.
        problem: Object containing `_ranges` and `_build_feature_types()` for `mixed_gower`.
        opt_passes (int): Number of optimization passes.
        memory_target_gb (float): Memory limit for batch processing.

    Returns:
        List[int]: Indices of selected diverse samples.
    """
    # Step 1: Constructive Algorithm - Initial Greedy Selection
    first_idx = np.argmax(y)  # Start with the highest weight
    indices = [first_idx]

    # Compute minimum weighted distances from all points to the first selected sample
    min_distances = min_weighted_distance(x_df, y, np.arange(len(x_df)), np.array([first_idx]), problem, memory_target_gb)

    for _ in range(num_samples - 1):
        best_idx = np.argmax(min_distances)  # Select the sample maximizing min distance
        indices.append(best_idx)

        # Update min distances
        new_distances = min_weighted_distance(x_df, y, np.arange(len(x_df)), np.array([best_idx]), problem, memory_target_gb)
        min_distances = np.minimum(min_distances, new_distances)

    # Step 2: Drop-Add Optimization (Fixed Iterations)
    for _ in range(MMDP_steps * num_samples):
        # Step 1: Drop the oldest selected point
        oldest_idx = indices.pop(0)

        # Update MinDist(x) efficiently
        drop_dists = min_weighted_distance(x_df, y, np.arange(len(x_df)), np.array([oldest_idx]), problem, memory_target_gb)
        needs_update = min_distances == drop_dists

        # Only recompute for affected points
        if np.any(needs_update):
            recompute_indices = np.where(needs_update)[0]
            min_distances[needs_update] = min_weighted_distance(x_df, y, recompute_indices, np.array(indices), problem, memory_target_gb)

        # Step 2: Select the best replacement point
        candidate_idx = np.argmax(min_distances)
        indices.append(candidate_idx)

        # Update MinDist(x) efficiently
        add_dists = min_weighted_distance(x_df, y, np.arange(len(x_df)), np.array([candidate_idx]), problem, memory_target_gb)
        min_distances = np.minimum(min_distances, add_dists)
    return indices
