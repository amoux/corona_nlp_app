from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import plac
import toml
import torch
from coronanlp.allenai import DownloadManager
from coronanlp.dataset import CORD19
from coronanlp.indexing import fit_index_ivf_hnsw
from coronanlp.ukplab import SentenceEncoder
from coronanlp.utils import save_stores

from utils import root_config, save_custom_stores

CONFIG = root_config()
CONFIG_FILE = './config.toml'
DEFAULT_DATADIR = './src/data/'
DATADIR = Path(DEFAULT_DATADIR).absolute()

DESC_UPDATE_CONFIG = """
Update the config toml file if `true` then the
following config key values will be updated in
default: `{0}`
=============================================
* DEFAULT CONFIGURATION KEYS ::::::::::::::::
=============================================
[engine][sents]
/path/to/..{5}store/<sents....>
[engine][index]
/path/to/..{5}store/<embed.npy>
[models][sentence_encoder] (default: {1})
[models][spacy_nlp]        (default: {2})
[cord][index_start]        (default: {3})
[cord][sort_first]         (default: {4})
==============================================
""".format(CONFIG_FILE,
           CONFIG['models']['sentence_encoder'],
           CONFIG['models']['spacy_nlp'],
           CONFIG['cord']['index_start'],
           CONFIG['cord']['sort_first'],
           DEFAULT_DATADIR)


@plac.annotations(
    sample=("Number of papers. '-1' for all papers", "option", "sample", int),
    minlen=("Minimum length of a string to consider", "option", "minlen", int),
    maxlen=("Maximum length of a string to consider", "option", "maxlen", int),
    index_start=("Starting index position for paper-ids, default: `1`",
                 "option", "index_start", int),
    workers=("Number of workers - same as; `cpu_count()`,",
             "option", "workers", int),
    sort_first=("Sort files before mapping", "option", "sort_first", bool),
    arch_date=(
        ("Use the an existing archive; e.g load source from date: `2020-03-20`"
         "Assumes, the archive's content exist; (replaces `source` argument)"),
        "option", "arch_date", str,
    ),
    source=("Path to cord19 dir of json files", "option", "source", str),
    encoder=("Pretrained model-name or path", "option", "encoder", str),
    max_length=("SentenceEncoder max_length for embeddings (default: 512)",
                "option", "max_length", int),
    batch_size=("SentenceEncoder batch_size (default; 12)",
                "option", "batch_size", int),
    nlp_model=("spaCy model name", "option", "nlp_model", str),
    update_config=(DESC_UPDATE_CONFIG, "option", "update_config", bool),
    config_file=(f"Main config, default: {CONFIG_FILE}",
                 "option", "config_file", str),
    store_name=(
        "Store name for all stores. (optional only when dirpath is used",
        "option", "store_name", str,
    ),
    dirpath=(
        ("Custom path/to/<store>/<mystore> to save `sents, index & embed` "
         "files Otherwise, the default directory will be used `./cache` "
         "NOTE: Windows need to use dirpath in order to save"),
        "option", "dirpath", str
    )
)
def main(
    sample: int = -1,
    minlen: int = 15,
    maxlen: int = 800,
    workers: int = 7,
    max_length: int = 512,
    batch_size: int = 12,
    index_start: Optional[int] = None,
    sort_first: Optional[bool] = None,
    arch_date: Optional[str] = None,
    source: Optional[Union[str, List[str]]] = None,
    nlp_model: Optional[str] = None,
    encoder: Optional[str] = None,
    update_config: bool = True,
    config_file: Optional[str] = None,
    store_name: Optional[str] = None,
    dirpath: Optional[str] = None,
):
    """Build and encode CORD-19 dataset texts to sentences and embed."""
    cfg_source: Union[str, List[str]]
    cfg_index_start: int
    cfg_sort_first: bool
    cfg_num_papers: int
    cfg_num_sents: int
    cfg_sents: str
    cfg_index: str  # faiss index db of the embed
    cfg_embed: str  # the actual embed <numpy.array>
    cfg_spacy_nlp: str
    cfg_encoder_model: str
    is_custom_store: bool
    config = CONFIG

    if dirpath is None:
        is_custom_store = False
    else:
        dirpath = Path(dirpath).absolute()
        if not dirpath.exists():
            dirpath.mkdir(parents=True)
        is_custom_store = True
    if not is_custom_store and store_name is None:
        raise ValueError(
            "When saving to default directory (cache) you need to provive "
            "the name the argument `store_name` is expected; please try "
            "again or ignore and use a custom directory. path instead. For "
            "this; the `dirpath` parameter is expected")

    arch = None
    if arch_date is not None:
        target_date = arch_date
        manager = DownloadManager()
        existing_dates = manager.all_archive_dates()
        if target_date in existing_dates:
            arch = manager.load_archive(target_date)
    if arch is not None:
        source = [p.as_posix() for p in arch.paths]

    if source is None:
        source = config['cord']['source']
    if index_start is None:
        index_start = config['cord']['index_start']
    if sort_first is None:
        sort_first = config['cord']['sort_first']
    if nlp_model is None:
        nlp_model = config['models']['spacy_nlp']
    if encoder is None:
        encoder = config['models']['sentence_encoder']

    cfg_spacy_nlp = nlp_model
    cfg_encoder_model = encoder
    cfg_source = source
    cfg_index_start = index_start
    cfg_sort_first = sort_first

    # SENTS::
    cord19 = CORD19(
        source=source,
        index_start=index_start,
        sort_first=sort_first,
        nlp_model=nlp_model
    )
    sample = cord19.sample(sample)
    sents = cord19.batch(sample, minlen, maxlen, workers)
    cfg_num_papers = sents.num_papers
    cfg_num_sents = sents.num_sents

    # EMBED::
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = SentenceEncoder.from_pretrained(encoder, device=device)
    if encoder.device.type == 'cuda':
        torch.cuda.empty_cache()
    embed = encoder.encode(
        sentences=sents,
        max_length=max_length,
        batch_size=batch_size,
        show_progress=True
    )

    # INDEX::
    N = embed.shape[0]
    nlist = int(np.sqrt(N))
    if N > 1_000_000:
        nlist = nlist ** 16
    index_ivf = fit_index_ivf_hnsw(embed, metric='l2', m=32)

    if is_custom_store:
        custom_fp = save_custom_stores(
            sents=sents,
            embed=embed,
            index=index_ivf,
            dirpath=dirpath
        )
        cfg_sents = custom_fp['sents']
        cfg_embed = custom_fp['embed']
        cfg_index = custom_fp['index']
    else:
        save_stores(
            sents=sents,
            embed=embed,
            index=index_ivf,
            store_name=store_name
        )
        cfg_sents = store_name
        cfg_embed = store_name
        cfg_index = store_name

    if is_custom_store:
        print(f'Done:  `sents, embed & index` saved in path: {dirpath}')
    else:
        print(f'Done: `sents, embed & index` saved to store: {store_name}')
    print("Updating configuration file with the updated settings.")

    if update_config:
        pass  # TODO: Updating config after building all stores does??

    config['models'].update({'sentence_encoder': cfg_encoder_model,
                             'spacy_nlp': cfg_spacy_nlp})
    config['stores'].update({'sents': cfg_sents,
                             'embed': cfg_embed,
                             'index': cfg_index,
                             'is_custom_store': is_custom_store})
    config['cord'].update({'source': cfg_source,
                           'index_start': cfg_index_start,
                           'sort_first': cfg_sort_first,
                           'num_papers': cfg_num_papers,
                           'num_sents': cfg_num_sents})
    if not is_custom_store and arch is not None:
        config['cord'].update({'metadata': arch.content['metadata'],
                               'version': arch.date,
                               'subsets': arch.source.names})

    file_path = CONFIG_FILE if config_file is None else config_file
    with open(file_path, 'w') as f:
        toml.dump(config, f)

    print(f"configuration file has been updated, file: {file_path}")


if __name__ == '__main__':
    plac.call(main)
