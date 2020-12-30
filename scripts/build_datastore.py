from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import plac
import toml
import torch
from coronanlp.allenai import DownloadManager
from coronanlp.dataset import CORD19
from coronanlp.indexing import fit_index_ivf_hnsw
from coronanlp.retrival import extract_titles_fast, tune_ids
from coronanlp.tasks import TaskList
from coronanlp.ukplab import SentenceEncoder
from coronanlp.utils import save_stores

from utils import CORONA_APP_CONFIG, app_config, save_custom_stores

CONFIG = app_config()
CONFIG_FILE =  Path(CORONA_APP_CONFIG).name
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


def server_sample(arch, cord19, min_title_len=8, return_mapping=False):
    # This method enforces that both titles from `metadata.csv`
    # and the actual titles extracted from the file can be
    # loaded (since the metadata.csv has bad UID's that do not
    # exist; in otherwords - it avoids issues when loading a title
    # or UID <--> PID from either sources. It also adds the benifit
    # of reducing the number of documents to tokenize. Since DEC-26
    # all versions from semantic scholar have over 29,000 articles!
    import pandas as pd
    df = pd.read_csv(arch.content['metadata'])
    df = df.loc[df['has_full_text'] == True, ['sha', 'title']]
    df.dropna(inplace=True)
    df.drop_duplicates(subset=['sha'], inplace=True)
    full_text = {uid: title for uid, title
                 in zip(df['sha'].tolist(), df['title'].tolist())
                 if len(title.strip().split()) > min_title_len}

    full_text_pids = sorted(cord19._encode(full_text.keys()))
    title_map = extract_titles_fast(cord19, full_text_pids, min_title_len)
    if return_mapping:
        return title_map
    sample = sorted(title_map.keys())
    return sample


def kaggle_sample(arch, cord19, encoder, min_title_len=8):
    mapping = server_sample(arch, cord19, min_title_len, return_mapping=True)
    kaggle_tasks = TaskList()
    target_size = len(mapping) // len(kaggle_tasks)
    tuned = tune_ids(encoder, mapping, kaggle_tasks, target_size)
    sample = sorted(set(tuned.pids))
    return sample


@plac.annotations(
    sample=("Number of papers. '-1' for all papers", "option", "sample", int),
    type=("Sample type; `server`, `kaggle` or `none` ", "option", "type", str),
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
    type: str = 'server',
    minlen: int = 15,
    maxlen: int = 600,
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

    sample_type = ''
    sample_types = ['server', 'kaggle', 'none']
    if type.lower().strip() in sample_types:
        sample_type = type.lower().strip()
    else:
        raise ValueError(
            f'Expected one sample type from: {sample_types}, '
            f'instead got: {type}. To do all sample pass `none`'
        )

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

    # EMBED::
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = SentenceEncoder.from_pretrained(encoder, device=device)
    if encoder.device.type == 'cuda':
        torch.cuda.empty_cache()

    paper_ids = []
    if arch is not None or sample_type != 'none':
        if sample_type == 'server':
            paper_ids = server_sample(arch, cord19)
        if sample_type == 'kaggle':
            paper_ids = kaggle_sample(arch, cord19, encoder)

        print('*{} sample reduced to: {:,}, from {:,} total.'.format(
            sample_type.title(), len(paper_ids), len(cord19)))
    else:
        paper_ids = cord19.sample(sample)

    sents = cord19.batch(paper_ids, minlen, maxlen, workers)
    embed = encoder.encode(
        sentences=sents,
        max_length=max_length,
        batch_size=batch_size,
        show_progress=True
    )
    cfg_num_papers = sents.num_papers
    cfg_num_sents = sents.num_sents

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
