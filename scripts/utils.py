from pathlib import Path
from typing import Dict, Union

import coronanlp
import faiss
import numpy
import toml


def root_config(fp='./config.toml'):
    abs_fp = Path(fp).absolute()
    return toml.load(abs_fp.as_posix())


def save_custom_stores(
    datadir: Union[str, Path],
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
        sents_fp_posix = sents_fp.absolute().as_posix()
        filepaths['sents'] = sents_fp_posix
        sents.to_disk(sents_fp_posix)

    if embed:
        embed_fp = datadir.joinpath(f"store_embed_{PIDS}.npy")
        embed_fp_posix = embed_fp.absolute().as_posix()
        filepaths['embed'] = embed_fp_posix
        numpy.save(embed_fp_posix, embed)

    if index:
        index_fp = datadir.joinpath(f"store_index_{PIDS}.bin")
        index_fp_posix = index_fp.absolute().as_posix()
        filepaths['index'] = index_fp_posix
        faiss.write_index(index, index_fp_posix)

    return filepaths
