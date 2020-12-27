from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import plac
import spacy
import toml
import torch
from coronanlp.core import SentenceStore
from coronanlp.retrival import common_tokens, extract_questions
from coronanlp.ukplab import SentenceEncoder
from coronanlp.utils import DataIO, split_on_char
from graphviz import Digraph

from utils import root_config

ROOT_SRC_DIR = './src/'
DEFAULT_CONFIG = './config.toml'


def build_qknn(
    sentences: List[str],
    encoder: SentenceEncoder,
    topk_ents: int = 30,
    nlist: int = 25,
    niter: int = 20,
    min_length: int = 4,
    max_length: int = 512,
    lowercase: bool = True,
    batch_size: int = 12,
    show_progress: bool = False,
    spacy_nlp: Optional[spacy.language.Language] = None,
):
    """"Build the QKNN data object for the web-application

    :param sentences: An iterable list of string sequences to use
        for clustering, kmeans and building dot-graph cats.
    :param nlist: Number of clusters.
    :param niter: Number of train iterations.
    :param topk_ents: Number of topk key_words.
    :param min_length: Token minimum length.
    """
    # cache the nlp object to avoid re-initialization within loop.
    nlp: spacy.language.Language
    if spacy_nlp is None:
        nlp = spacy.load('en_core_web_sm')
    else:
        nlp = spacy_nlp

    # Build the entities based on freq per centroid
    def extract_entities(x):
        return common_tokens(x, min_length, lowercase, nlp)

    # Extract and encode questions for clustering.
    embeddings = encoder.encode(
        sentences, max_length, batch_size, show_progress)

    # Build kmeans | sort questions in relation to knn distances
    n = len(sentences)
    d = embeddings.shape[1]
    kmeans = faiss.Kmeans(d, nlist, niter=niter, verbose=True)
    kmeans.train(embeddings)
    index = faiss.IndexFlat(d)
    index.add(embeddings)

    topk = n // nlist
    _, centroids = index.search(kmeans.centroids, topk)
    search_nlist = centroids.shape[0]

    clusters: List[List] = [[] for _ in range(search_nlist)]
    for k in range(search_nlist):
        for nn in centroids[k]:
            clusters[k].append(sentences[nn])
    print(f'(centroids, neighbors) : {centroids.shape}')

    cats = {}
    nn = centroids.shape[1]
    for k in range(search_nlist):
        toks = extract_entities(clusters[k])
        ents = toks[:nn - 1 if nn % 2 else nn]
        if k not in cats:
            cats[k] = ents

    # Display a preview of the categories
    for k in cats:
        category, entities = cats[k][0],  cats[k][1:6]
        print(f'{category}\t-> {entities}')

    Q, K = {}, []
    for centroid in cats:
        root = cats[centroid][0][0]
        ents = cats[centroid][1: topk_ents + 1]
        ents.sort(key=lambda k: k[1], reverse=True)
        ents, _ = zip(*ents)
        K.append(list(ents))
        if root not in Q:
            Q[root] = clusters[centroid]
        else:
            Q[root].extend(clusters[centroid])

    qknn = {q: {'key_words': nn, 'questions': k}
            for (q, k), nn in zip(Q.items(), K)}

    return {'qknn': qknn, 'cats': cats}


def qknn_to_diagraph(data, return_edges=False):
    cats: Dict[int, List[Tuple[str, int]]] = {}
    if isinstance(data, dict):
        if 'cats' in data:
            cats = data['cats']
        else:
            cats = data

    edges = []
    for cat in cats:
        common = cats[cat]
        for i in range(0, len(common), 2):
            x = common[i: min(i + 2, len(common))]
            nodes, k = zip(*x)
            edges.append(nodes)

    graph = Digraph()
    for tail, head in edges:
        graph.edge(tail, head)

    if return_edges:
        return graph, edges
    else:
        return graph


@plac.annotations(
    nlist=("Number of clusters", "option", "nlist", int),
    niter=("Number of iterations", "option", "niter", int),
    minlen=("Minimum length to consider a word", "option", "minlen", int),
    topk_ents=("Number of topk key_words", "option", "topk_ents", int),
    store_name=("Store name, otherwise from config", "option", "store", str),
    diagraph=("Build the dot diagram of QKNN", "option", "diagraph", bool),
    qknn_fp=("File path to save QKNN", "option", "qknn_fp", str),
    graph_fp=("File path where to save the PDF", "option", "graph_fp", str),
)
def main(nlist: int = 10,
         niter: int = 20,
         minlen: int = 3,
         topk_ents: int = 10,
         store_name: Optional[str] = None,
         diagraph: bool = True,
         qknn_fp: Optional[str] = None,
         graph_fp: Optional[str] = None):
    """Build the QKNN data object for the web-application."""

    root = Path(ROOT_SRC_DIR).absolute()
    if qknn_fp is None:
        qknn_fp = root.joinpath('QKNN.pkl')
    if graph_fp is None:
        graph_fp = root.joinpath('qknn_graph.gv')

    config = root_config()
    if store_name is None:
        store_name = config['stores']['sents']

    model_name = config['models']['sentence_encoder']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    encoder = SentenceEncoder.from_pretrained(model_name, device=device)
    sents = SentenceStore.from_disk(store_name)
    questions = extract_questions(sents, sents.min(), remove_empty=True)

    flatten = []
    for question in questions:
        if question.count('?') > 1:
            splits = split_on_char(question, char='?')
            flatten.extend(splits)
        else:
            flatten.append(question)

    output = build_qknn(sentences=flatten,
                        encoder=encoder,
                        topk_ents=topk_ents,
                        min_length=minlen,
                        nlist=nlist,
                        niter=niter)

    DataIO.save_data(qknn_fp, output['qknn'])
    config['streamlit']['qknn_file'] = qknn_fp.as_posix()
    with open(DEFAULT_CONFIG, 'w') as f:
        toml.dump(config, f)
    print('Configuration was updated, automatically '
          'loaded when the web-app is initialized!')

    if diagraph:
        graph = qknn_to_diagraph(output)
        try:
            graph.render(graph_fp.as_posix(), view=False)
        except Exception as e:
            print(
                'It looks like graphviz is installed but missing dependencies.'
                f' However, QKNN object is saved & the config updated.\n\t{e}')
        else:
            print(f'Done! QKNN object saved in {qknn_fp}')


if __name__ == '__main__':
    plac.call(main)
