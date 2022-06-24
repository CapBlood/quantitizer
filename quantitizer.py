import numpy as np
from scipy.cluster.vq import kmeans2

from pq_array import PQ


# http://ethen8181.github.io/machine-learning/deep_learning/multi_label/product_quantization.html
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

    code_books = np.array(code_books)
    indexes = np.array(indexes).T

    # return code_books, np.array(indexes).T
    return PQ(
        len(vectors), len(vectors[0]), sub_size,
        n_cluster, indexes, code_books)


# TODO: не работает
# https://github.com/avidale/compress-fasttext/blob/master/compress_fasttext/quantization.py
def quantitize_experimental(vectors, sub_size=8, sample=None, n_cluster=256, n_iter=20, minit='points', seed=123):
    encoder = pqkmeans.encoder.PQEncoder(
        iteration=n_iter,
        num_subdim=sub_size,
        Ks=n_cluster
    )

    if sample is None:
        selection = vectors
    else:
        indexes = np.random.randint(vectors, size=sample)
        selection = vectors[indexes]

    encoder.fit(selection)
    indexes = encoder.transform(vectors).astype(PQ.index_type(n_cluster))
    codes = encoder.codewords

    return PQ(
        len(vectors), len(vectors[0]), sub_size,
        n_cluster, indexes, codes)
