from pathlib import Path

import faiss
import numpy as np
import plac
from corona_nlp.dataset import CORD19Dataset
from corona_nlp.transformer import SentenceTransformer

DEFAULT_SOURCE = [
    '/home/carlos/Datasets/CORD-19-research-challenge/comm_use_subset/comm_use_subset/pdf_json',
    '/home/carlos/Datasets/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/pdf_json',
    '/home/carlos/Datasets/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/pdf_json'
]


@plac.annotations(
    num_papers=("Number of papers. '-1' for all papers", "option", "n", int),
    minlen=("Minimum length of a string to consider", "option", "minlen", int),
    sort_first=("Sort files before mapping 0:false, 1:true",
                "option", "sort", int),
    nlp_model=("spaCy model name", "option", "m", str),
    data_dir=("Path to the directory for outputs", "option", "data_dir", str),
    source=("Path to cord19 dir of json files", "option", "source", str),
    encoder=("Path to sentence encoder model", "option", "encoder", str),
)
def main(num_papers=-1, minlen=20, sort_first=0, nlp_model="en_core_sci_sm",
         data_dir="data/", source=None, encoder="model/scibert-nli"):
    """Build and encode CORD-19 dataset texts to sentences and embeddings."""
    if source is None:
        source = DEFAULT_SOURCE
    if sort_first == 0:
        sort_first = False
    else:
        sort_first = True

    data_dir = Path(data_dir)
    if not data_dir.exists():
        data_dir.mkdir(parents=True)

    cord19 = CORD19Dataset(source=source,
                           text_keys=("body_text",),
                           sort_first=sort_first,
                           nlp_model=nlp_model)

    sample = cord19.sample(num_papers)
    papers = cord19.batch(sample, minlen=minlen)

    # save the instance of papers to file
    sents_file = data_dir.joinpath(f"sents_{papers.num_papers}.pkl")
    papers.to_disk(sents_file)

    encoder = SentenceTransformer(encoder)
    embedding = encoder.encode(papers, show_progress=True)
    assert embedding.shape[0] == len(papers)

    # save the encoded embeddings to file
    embed_file = data_dir.joinpath(f"embed_{papers.num_papers}.npy")
    np.save(embed_file, embedding)

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
    index_file = data_dir.joinpath(f'IVF{nlist}_HNSW{m}_NP{npapers}.bin')
    faiss.write_index(index_ivf, index_file.as_posix())

    print(f'Done: index and papers saved in path: {data_dir}')


if __name__ == '__main__':
    plac.call(main)