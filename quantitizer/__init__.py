import numpy as np
from scipy.cluster.vq import kmeans2
from cuml import KMeans
from cuml.cluster import KMeans

from quantitizer._pq_array import PQ


def quantitize(vectors, sub_size=8, n_cluster=256, n_iter=20, minit='points', seed=123):
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


def quantitize_experimental(vectors, sub_size=8, sample=None, n_cluster=256, n_iter=20, minit='points', seed=123):
    if len(vectors[0]) % sub_size != 0:
        raise Exception(f"sub_size должен нацело делить {len(vectors[0])}")

    np.random.seed(seed)
    parts = np.array_split(vectors, sub_size, axis=1)

    code_books, indexes = [], []
    for part in parts:
        # centroid, label = kmeans2(
        #     part, n_cluster, n_iter, minit=minit, seed=seed)
        kmeans_float = KMeans(
            n_clusters=n_cluster, max_iter=10)
        kmeans_float.fit(part)
        code_books.append(kmeans_float.cluster_centers_.tolist())
        indexes.append(kmeans_float.labels_.tolist())

    code_books = np.array(code_books)
    indexes = np.array(indexes).T

    # return code_books, np.array(indexes).T
    return PQ(
        len(vectors), len(vectors[0]), sub_size,
        n_cluster, indexes, code_books)
