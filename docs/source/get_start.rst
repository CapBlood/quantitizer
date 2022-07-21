Getting started
======================
quantitizer allows to compress gensim KeyedVectors models (word2vec, for example) and fasttext models by quantitizing
them matrixes.

If you want to load precompressed models you can use the next example:

.. code-block:: python

    from quantitizer.pretrain import load
    from quantitizer.integration.gensim.fasttext import load_ft

    load("fasttext-compressed-en-100")
    ft = load_ft("fasttext_compressed_en_100")
    vec = ft.get_vector("word")

if you want to quantitize matrix you can use the next code:

.. code-block:: python

    import numpy as np

    from quantitizer import quantitize

    matrix = np.random.random((50000, 1000))
    qmatrix = quantitize(matrix, sub_size=5)
    matrix.nbytes // 1024 // 1024 # В мегабайтах
    >> 381

    qmatrix.nbytes // 1024 // 1024
    >> 2

If you want to compress KeyedVectors model you can use :py:func:`quantitizer.integration.gensim.word2vec.quantitize_wv`
and :py:func:`quantitizer.integration.gensim.fasttext.quantitize_ft` for Fasttext.

