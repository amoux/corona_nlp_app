import os
from pathlib import Path
from typing import Any, Dict, MutableMapping, Optional, Union

import coronanlp
import faiss
import numpy
import toml

CORONA_APP_CONFIG = os.environ.get("CORONA_APP_CONFIG")


def app_config(toml_config: Optional[str] = None) -> MutableMapping[str, Any]:
    if toml_config is None:
        toml_config = CORONA_APP_CONFIG
    if toml_config is None:
        raise Exception(
            "CORONA_APP_CONFIG cannot be empty, make sure to "
            "export CORONA_APP_CONFIG='path/to/my_config.toml'")
    config_fp = Path(toml_config).absolute()
    if not config_fp.is_file():
        raise Exception(
            f"Expected valid TOML config file, got: {config_fp.as_posix()}")
    config_dict = toml.load(config_fp.as_posix())
    return config_dict


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
