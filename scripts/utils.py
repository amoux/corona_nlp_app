from typing import Dict
from pathlib import Path

import coronanlp
import faiss
import numpy
import toml

CONFIG_FILE = './config.toml'
CONFIG_DICT = toml.load(Path(CONFIG_FILE).absolute().as_posix())


def save_custom_stores(
    datadir: str,
    sents: coronanlp.SentenceStore,
    embed: numpy.ndarray,
    index: faiss.Index,
) -> Dict[str, str]:

    datadir = Path(datadir).absolute()
    if not datadir.exists():
        datadir.mkdir(parents=True)

    PIDS = sents.num_papers
    filepaths = {}

    if sents:
        sents_fp = datadir.joinpath(f"store_sents_{PIDS}.pkl")
        sents_fp = sents_fp.absolute().as_posix()
        filepaths['sents'] = sents_fp
        sents.to_disk(sents_fp)

    if embed:
        embed_fp = datadir.joinpath(f"store_embed_{PIDS}.npy")
        embed_fp = embed_fp.absolute().as_posix()
        filepaths['embed'] = embed_fp
        numpy.save(embed_fp, embed)

    if index:
        index_fp = datadir.joinpath(f"store_index_{PIDS}.bin")
        index_fp = index_fp.absolute().as_posix()
        filepaths['index'] = index_fp
        faiss.write_index(index, index_fp)

    return filepaths
