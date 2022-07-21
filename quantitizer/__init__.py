import numpy as np
from scipy.cluster.vq import kmeans2

from quantitizer._pq_array import PQ


def quantitize(vectors: np.ndarray, sub_size: int = 8,
               n_cluster: int = 256, n_iter: int = 20, minit: str = 'points',
               seed: int = 123) -> PQ:
    """Quantitizes matrixes.

    Parameters
    ----------
    vectors : np.ndarray
        Matrix to quantitize.
    sub_size : int, default: 8
        Number of partitions to quantitize.
    n_cluster: int
        Number of clusters to quantitize each part.
    n_iter : int, default: 20
        Number of iterations. It affects the accuracy of calculations.
    minit : str
        Method to initialize from scipy.
    seed : int
        Seed for kmeans2.


    Returns
    -------
    PQ
        Quantitized matrix.
    """

    if len(vectors[0]) % sub_size != 0:
        raise Exception(f"sub_size должен нацело делить {len(vectors[0])}")

    np.random.seed(seed)
    parts = np.array_split(vectors, sub_size, axis=1)

    code_books, indexes = [], []
    for part in parts:
        centroid, label = kmeans2(
            part, n_cluster, n_iter, minit=minit, seed=seed)
        code_books.append(centroid)
        indexes.append(label)

    code_books = np.array(code_books, dtype=np.float16)
    indexes = np.array(indexes, dtype=np.int8).T

    # return code_books, np.array(indexes).T
    return PQ(
        len(vectors), len(vectors[0]), sub_size,
        n_cluster, indexes, code_books)
