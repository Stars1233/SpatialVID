import numpy as np
from scipy.sparse import csr_matrix

def expand(loaded_data):
    reconstructed_sparse_matrices = []
    num_frames = (len(loaded_data) - 1) // 3
    matrix_shape = loaded_data['shape']

    for i in range(num_frames):
        data = loaded_data[f'f_{i}_data']
        indices = loaded_data[f'f_{i}_indices']
        indptr = loaded_data[f'f_{i}_indptr']
        # reconstruction csr_matrix
        reconstructed_matrix = csr_matrix((data, indices, indptr), shape=matrix_shape)
        reconstructed_sparse_matrices.append(reconstructed_matrix)

    reconstructed_mask_3d = np.stack([m.toarray() for m in reconstructed_sparse_matrices], axis=0)
    # reconstructed_mask_3d.shape
    return reconstructed_mask_3d
