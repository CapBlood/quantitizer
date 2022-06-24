import copy

import fasttext.util
import numpy as np
from gensim.models import FastText
from gensim.models.fasttext import save_facebook_model

if __name__ == "__main__":
    # fasttext.util.download_model('en', if_exists='ignore')
    # ft = fasttext.load_model('cc.en.300.bin')
    ft = FastText.load_fasttext_format("../data/cc.en.300.bin")
    print(ft.wv.get_vector("life"))

    # new_ft = copy.deepcopy(ft)
    # new_ft.wv.vectors_ngrams = new_ft.wv.vectors_ngrams.astype(np.float16)
    # new_ft.wv.vectors_vocab = new_ft.wv.vectors_vocab.astype(np.float16)
    # save_facebook_model(new_ft, "../out/compressed_ft.bin")
