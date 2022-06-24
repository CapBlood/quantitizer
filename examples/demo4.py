from gensim.models import FastText

if __name__ == "__main__":
    ft = FastText.load_fasttext_format("../out/compressed_ft.bin")
    print(ft.wv.get_vector("life"))
