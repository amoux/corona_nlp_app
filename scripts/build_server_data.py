from pathlib import Path
from typing import List, Optional, Union

import faiss
import numpy as np
import plac
import toml
from corona_nlp.dataset import CORD19Dataset
from corona_nlp.transformer import SentenceTransformer

DEFAULT_CONFIG = './config.toml'
DEFAULT_DATADIR = './src/data/'
CONFIG = toml.load(Path(DEFAULT_CONFIG).absolute().as_posix())
DATADIR = Path(DEFAULT_DATADIR).absolute()

DESC_UPDATE_CONFIG = """
Update the config toml file if `true` then the
following config key values will be updated in
default: `{0}`
=============================================
* DEFAULT CONFIGURATION KEYS ::::::::::::::::
=============================================
[engine][papers]
/path/to/..{5}/store/<sentences...pkl >
[engine][index]
/path/to/..{5}/store/<embedding...bin >
[models][sentence_transformer] (default: {1})
[models][spacy_nlp]            (default: {2})
[cord.init][index_start]       (default: {3})
[cord.init][sort_first]        (default: {4})
==============================================
""".format(DEFAULT_CONFIG,
           CONFIG['models']['sentence_transformer'],
           CONFIG['models']['spacy_nlp'],
           CONFIG['cord']['init']['index_start'],
           CONFIG['cord']['init']['sort_first'],
           DEFAULT_DATADIR)


@plac.annotations(
    sample=("Number of papers. '-1' for all papers", "option", "sample", int),
    minlen=("Minimum length of a string to consider", "option", "minlen", int),
    index_start=("Starting index position for paper-ids, default: `1`",
                 "option", "index_start", int),
    sort_first=("Sort files before mapping", "option", "sort_first", bool),
    source=("Path to cord19 dir of json files", "option", "source", str),
    encoder=("Path to sentence encoder model", "option", "encoder", str),
    nlp_model=("spaCy model name", "option", "nlp_model", str),
    update_config=(DESC_UPDATE_CONFIG, "option", "update_config", bool),
    config_file=(f"Main config, default: {DEFAULT_CONFIG}",
                 "option", "config_file", str),
    data_dir=("Default data directory, default: `./src/data/`",
              "option", "data_dir", str)
)
def main(
    sample: int = -1,
    minlen: int = 25,
    index_start: Optional[int] = None,
    sort_first: Optional[bool] = None,
    source: Optional[Union[str, List[str]]] = None,
    nlp_model: Optional[str] = None,
    encoder: Optional[str] = None,
    update_config: bool = True,
    config_file: Optional[str] = None,
    data_dir: Optional[str] = None,
):
    """Build and encode CORD-19 dataset texts to sentences and embeddings."""
    cfg_source: Union[str, List[str]]
    cfg_index_start: int
    cfg_sort_first: bool
    cfg_num_papers: int
    cfg_num_sentences: int
    cfg_papers: str
    cfg_index: str  # faiss index db of the embeddings
    cfg_numpy: str  # the actual embeddings <numpy.array>
    cfg_spacy_nlp: str
    cfg_encoder_model: str

    config = CONFIG
    if data_dir is None:
        data_dir = DATADIR
    else:
        data_dir = Path(data_dir).absolute()
    if not data_dir.exists():
        data_dir.mkdir(parents=True)

    if source is None:
        source = config['cord']['source']
    if index_start is None:
        index_start = config['cord']['init']['index_start']
    if sort_first is None:
        sort_first = config['cord']['init']['sort_first']
    if nlp_model is None:
        nlp_model = config['models']['spacy_nlp']
    if encoder is None:
        encoder = config['models']['sentence_transformer']

    cfg_spacy_nlp = nlp_model
    cfg_encoder_model = encoder
    cfg_source = source
    cfg_index_start = index_start
    cfg_sort_first = sort_first

    cord19 = CORD19Dataset(source=source,
                           index_start=index_start,
                           sort_first=sort_first,
                           nlp_model=nlp_model,
                           text_keys=('body_text',),)

    sample = cord19.sample(sample)
    papers = cord19.batch(sample, minlen=minlen)
    cfg_num_papers = papers.num_papers
    cfg_num_sentences = papers.num_sents

    # save the instance of papers to file
    papers_file = data_dir.joinpath(
        f"STRING_STORE_{papers.num_papers}.pkl"
    )
    papers.to_disk(papers_file)
    cfg_papers = papers_file.absolute().as_posix()

    encoder = SentenceTransformer(encoder)
    embedding = encoder.encode(papers, show_progress=True)
    assert embedding.shape[0] == len(papers)

    # save the encoded embeddings to file
    embed_file = data_dir.joinpath(
        f"EMBED_STORE_{papers.num_papers}.npy"
    )
    np.save(embed_file, embedding)
    cfg_numpy = embed_file.absolute().as_posix()

    m = 32
    n, d = embedding.shape
    nlist = int(np.sqrt(n))
    quantizer = faiss.IndexHNSWFlat(d, m)
    index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    index_ivf.verbose = True
    index_ivf.train(embedding)
    index_ivf.add(embedding)
    assert index_ivf.ntotal == embedding.shape[0]

    # save the indexer of embeddings to file
    npapers = papers.num_papers
    index_file = data_dir.joinpath(
        f'EMBED_STORE_IVF{nlist}_HNSW{m}_NP{npapers}.bin'
    )
    faiss.write_index(index_ivf, index_file.as_posix())
    cfg_index = index_file.absolute().as_posix()

    print(f'Done: index and papers saved in path: {data_dir}')
    print("Updating configuration file with the updated settings.")

    if update_config:
        config['cord'].update({'source': cfg_source})
        config['cord']['init'].update({'index_start': cfg_index_start,
                                       'sort_first': cfg_sort_first})
        config['streamlit'].update({'num_papers': cfg_num_papers,
                                    'num_sentences': cfg_num_sentences})
        config['engine'].update({'index': cfg_index,
                                 'papers': cfg_papers})
        config['data'].update({'embeddings': cfg_numpy})
        config['models'].update({'sentence_transformer': cfg_encoder_model,
                                 'spacy_nlp': cfg_spacy_nlp})

    file_path = DEFAULT_CONFIG if config_file is None else config_file
    with open(file_path, 'w') as f:
        toml.dump(config, f)

    print(f"configuration file has been updated, file: {file_path}")


if __name__ == '__main__':
    plac.call(main)
