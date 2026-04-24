import numpy as np
import faiss
import multiprocessing

def solve(base_vectors, query_vectors, k, K, time_budget):
    """
    Finds the top K most representative base vectors for a given query population.
    Blends zero-copy memory heuristics with mathematically perfect tie-breaking.
    """
    N, d = base_vectors.shape

    faiss.omp_set_num_threads(multiprocessing.cpu_count())

    if base_vectors.dtype != np.float32 or not base_vectors.flags['C_CONTIGUOUS']:
        base_vectors = np.ascontiguousarray(base_vectors, dtype=np.float32)
    if query_vectors.dtype != np.float32 or not query_vectors.flags['C_CONTIGUOUS']:
        query_vectors = np.ascontiguousarray(query_vectors, dtype=np.float32)

    index = faiss.IndexFlatL2(d)
    index.add(base_vectors)
    
    _, indices = index.search(query_vectors, k)

    flat_indices = indices.ravel()
    valid_indices = flat_indices[flat_indices >= 0]
    
    counts = np.bincount(valid_indices, minlength=N)

    idx_array = np.arange(N, dtype=np.int64)
    sorted_indices = np.lexsort((idx_array, -counts))

    return sorted_indices[:K]
