import numpy as np

from quantitizer import quantitize, quantitize_experimental

if __name__ == "__main__":
    path_vectors = "data/181/model.model.vectors.npy"
    path_ngrams = "data/181/model.model.vectors_ngrams.npy"

    vectors = np.load(path_vectors)
    ngrams = np.load(path_ngrams)

    print(f"Размер до: {ngrams.nbytes // 1024 // 1024} Мб")

    qvectors = quantitize(ngrams, sub_size=2)
    print(f"Размер после: {qvectors.nbytes // 1024 // 1024} Mб")
