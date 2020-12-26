from pathlib import Path
from typing import List, Optional

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
from spacy.language import Language

from utils import root_config

ROOT_SRC_DIR = './src/'
DEFAULT_CONFIG = './config.toml'


def build_qknn(
    sentences: List[str],
    encoder: SentenceEncoder,
    spacy_nlp: Language,
    max_length: int = 500,
    min_length: int = 4,
    lowercase: bool = False,
    batch_size: int = 9,
    nlist: int = 25,
    niter: int = 20,
    top_k_ents: int = 30,
):
    """"Build the QKNN data object for the web-application

    :param sentences: An iterable list of string sequences to use
        for clustering, kmeans and building dot-graph cats.
    :param nlist: Number of clusters.
    :param niter: Number of train iterations.
    :param top_k_ents: Number of topk key_words.
    :param min_length: Token minimum length.
    """
    # Extract and encode questions for clustering.
    embeddings = encoder.encode(sentences, max_length, batch_size)

    # Build kmeans | sort questions in relation to knn distances
    n = len(sentences)
    topk = n // nlist
    ndim = embeddings.shape[1]
    kmeans = faiss.Kmeans(ndim, nlist, niter=niter, verbose=True)
    kmeans.train(embeddings)
    index = faiss.IndexFlat(ndim)
    index.add(embeddings)

    _, I = index.search(kmeans.centroids, topk)
    cluster = [[] for _ in range(I.shape[0])]
    for k in range(I.shape[0]):
        for nn in I[k]:
            cluster[k].append(sentences[nn])
    print(f'(centroids, neighbors) : {I.shape}')

    # Build the entities based on freq per centroid
    def extract_entities(x):
        return common_tokens(x, min_length, lowercase, spacy_nlp)

    cats = {}
    nn = I.shape[1]
    for k in range(I.shape[0]):
        toks = extract_entities(cluster[k])
        ents = toks[:nn - 1 if nn % 2 else nn]
        if k not in cats:
            cats[k] = ents

    # Display a preview of the categories
    for k in cats:
        category = cats[k][0]
        entities = cats[k][1:6]
        print(f'{category}\t-> {entities}')

    Q, K = {}, []
    for centroid in cats:
        root = cats[centroid][0][0]
        ents = cats[centroid][1: top_k_ents + 1]
        ents.sort(key=lambda k: k[1], reverse=True)
        ents, _ = zip(*ents)
        K.append(list(ents))

        if root not in Q:
            Q[root] = cluster[centroid]
        else:
            Q[root].extend(cluster[centroid])

    qknn = {q: {'key_words': nn, 'questions': k}
            for (q, k), nn in zip(Q.items(), K)}

    return {'qknn': qknn, 'cats': cats}


def build_graph(data):
    edges = []
    cats = data['cats']
    for cat in cats:
        common = cats[cat]
        for i in range(0, len(common), 2):
            x = common[i: min(i + 2, len(common))]
            nodes, k = zip(*x)
            edges.append(nodes)
    graph = Digraph()
    for tail, head in edges:
        graph.edge(tail, head)
    return graph, edges


@plac.annotations(
    nlist=("Number of clusters", "option", "nlist", int),
    niter=("Number of iterations", "option", "niter", int),
    minlen=("Length of the questions extracted", "option", "minlen", int),
    top_k_ents=("Number of topk key_words", "option", "topk_ents", int),
    store_name=("Store name, otherwise from config", "option", "store", str),
    qknn_fp=("File path to save QKNN", "option", "qknn_fp", str),
    graph_fp=("File path where to save the PDF", "option", "graph_fp", str),
)
def main(nlist: int = 10,
         niter: int = 20,
         minlen: int = 30,
         top_k_ents: int = 10,
         store_name: Optional[str] = None,
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

    spacy_nlp = spacy.load('en_core_web_sm')
    encoder = SentenceEncoder.from_pretrained(model_name, device=device)
    sents = SentenceStore.from_disk(store_name)
    questions = extract_questions(sents, minlen, remove_empty=True)

    flatten = []
    for question in questions:
        if question.count('?') > 1:
            splits = split_on_char(question, char='?')
            flatten.extend(splits)
        else:
            flatten.append(question)

    output = build_qknn(sentences=flatten,
                        encoder=encoder,
                        spacy_nlp=spacy_nlp,
                        max_length=512,
                        min_length=minlen,
                        lowercase=True,
                        batch_size=9,
                        nlist=nlist,
                        niter=niter,
                        top_k_ents=top_k_ents)

    QKNN = output['qknn']
    graph, _ = build_graph(output)
    graph.render(graph_fp.as_posix(), view=False)

    DataIO.save_data(qknn_fp, QKNN)
    print(f'Done! QKNN object saved in {qknn_fp}')

    config['streamlit']['qknn_file'] = qknn_fp.as_posix()

    with open(DEFAULT_CONFIG, 'w') as f:
        toml.dump(config, f)

    print('Configuration was updated, automatically '
          'loaded when the web-app is initialized!')


if __name__ == '__main__':
    plac.call(main)
