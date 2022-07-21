from mega import Mega

from quantitizer.exceptions import NotFoundModel

__mega = Mega()
__url_map = dict({
    "fasttext-compressed-en-100":
        "https://mega.nz/file/KyA2UDZK#dMbytXDGGiUdIbJyx64OpS3sZKlA2dUkExe5bGbAAkA",
    "w2v-compressed-en-100":
        "https://mega.nz/file/3jQRVRha#gibpsoGJPyPyB93In1ZMk451WWy6vGuZMRO9y4_SbxE"
})


def load(name: str, outdir: str = ".") -> None:
    """Loads pretrain models.

    Parameters
    ----------
    name : str
        Name of pretrain model.
    outdir : str
        Path to load.
    """

    if name not in __url_map:
        raise NotFoundModel("Model not found")

    __mega.download_url(__url_map[name], dest_path=outdir)
