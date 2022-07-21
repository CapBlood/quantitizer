from typing import Literal, Union, Type

import numpy as np
from gensim.models import KeyedVectors

from quantitizer import quantitize, PQ

MODE_QUANTITIZE = Literal['cpu', 'cuda']


def quantitize_wv(wv: KeyedVectors, sub_size: int,
                  mode: MODE_QUANTITIZE = "cpu", n_iter: int = 5) -> KeyedVectors:
    """Compresses KeyedVectors model.

    Parameters
    ----------
    wv : KeyedVectors
        KeyedVectors model to compress it.
    sub_size : int
        Number of partitions to compress.
    mode : {'cuda', 'cpu'}, default: 'cpu'
        Calculation mode. It may be 'cuda' or 'cpu'.
    n_iter : int, default: 5
        Number of iterations. It affects the accuracy of calculations.

    Returns
    -------
    KeyedVectors
        Compressed KeyedVectors model.
    """

    if mode == "cpu":
        wv_q: KeyedVectors = make_new_wv_model(
            wv, quantitize(wv.vectors, sub_size=sub_size, n_iter=n_iter))
    elif mode == "cuda":
        from quantitizer.cuda import quantitize_cuda

        wv_q: KeyedVectors = make_new_wv_model(
            wv, quantitize_cuda(wv.vectors, sub_size=sub_size, n_iter=n_iter))
    else:
        raise ValueError("mode must be 'cpu' or 'cuda'")

    return wv_q


def load_wv(path: str) -> KeyedVectors:
    """Loads compressed KeyedVectors model from local disk.

    Parameters
    ----------
    path : str
        Path to compressed KeyedVectors model to load.

    Returns
    -------
    KeyedVectors
        Compressed KeyedVectors model.
    """

    wv = KeyedVectors.load(path)
    return wv


def make_new_wv_model(
        wv: KeyedVectors,
        new_vectors: Union[np.ndarray, PQ],
        new_vocab: dict = None,
        cls: Type[KeyedVectors] = None,
):
    cls = cls or KeyedVectors

    new_wv: cls = cls(
        vector_size=wv.vector_size
    )
    new_wv.vectors_vocab = None  # if we don't fine tune the model we don't need these vectors
    new_wv.vectors = new_vectors  # quantized vectors top_vectors
    if new_vocab is None:
        new_wv.key_to_index = wv.key_to_index
    else:
        new_wv.key_to_index = new_vocab

    if hasattr(new_wv, 'update_index2word'):
        new_wv.update_index2word()

    return new_wv
