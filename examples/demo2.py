import numpy as np

from quantitizer import quantitize, quantitize_cuda


if __name__ == "__main__":
    matrix = np.random.random((50000, 1000))

    print(f"Размер до: {matrix.nbytes // 1024 // 1024} Мб")

    qmatrix = quantitize(matrix, sub_size=5)
    print(f"Размер после: {qmatrix.nbytes // 1024 // 1024} Mб")
