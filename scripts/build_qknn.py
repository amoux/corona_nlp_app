from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import graphviz as graphviz
import plac
import toml
from corona_nlp.datatypes import Papers
from corona_nlp.retrival import common_tokens, extract_questions
from corona_nlp.transformer import SentenceTransformer
from corona_nlp.utils import DataIO

from utils import CONFIG_DICT

ROOT_SRC_DIR = './src/'
DEFAULT_CONFIG = './config.toml'


def build_terminology_graph(
        out_file: str = 'qknn_graph.gv',
        cats: Dict[int, List[Tuple[str, int]]] = None,
) -> None:
    pairs = 2
    edges = []
    for cat in cats:
        common = cats[cat]
        maxlen = len(common)
        for i in range(0, maxlen, pairs):
            x = common[i: min(i + pairs, maxlen)]
            nodes, k = zip(*x)
            edges.append(nodes)

    graph = graphviz.Digraph()
    for tail, head in edges:
        graph.edge(tail, head)

    graph.render(out_file, view=False)


def build_qknn(
    papers_filepath: str,
    nlist: int = 10,
    niter: int = 20,
    minlen: int = 30,
    num_topk_ents: int = 10,
    terminology_graph: bool = False,
    out_graph_pdf: str = 'clustered_questions.gv',
):
    # extract and encode questions for clustering.
    papers = Papers.from_disk(papers_filepath)
    questions = extract_questions(papers, minlen)

    encoder = SentenceTransformer(
        model_path=CONFIG_DICT['models']['sentence_transformer']
    )
    embedding = encoder.encode(questions)

    topk = len(questions) // nlist
    ndim = embedding.shape[1]

    # build kmeans
    kmeans = faiss.Kmeans(ndim, nlist, niter=niter, verbose=True)
    kmeans.train(embedding)

    # build a basic index flat
    index = faiss.IndexFlat(ndim)
    index.add(embedding)
    D, I = index.search(kmeans.centroids, topk)

    # sort questions in relation to knn distances
    cluster = [[] for _ in range(I.shape[0])]
    for k in range(I.shape[0]):
        for nn in I[k]:
            cluster[k].append(questions[nn])

    print(f'(centroids, neighbors) : {I.shape}')

    # build the entities based on freq per centroid
    nn = I.shape[1]
    cats = {}
    for k in range(I.shape[0]):
        toks = common_tokens(cluster[k])
        ents = toks[:nn - 1 if nn % 2 else nn]
        if k not in cats:
            cats[k] = ents

    # display a preview of the categories
    for k in cats:
        category = cats[k][0]
        entities = cats[k][1:6]
        print(f'{category}\t-> {entities}')

    if terminology_graph:
        build_terminology_graph(out_graph_pdf, cats=cats)

    Q, K = {}, []
    for centroid in cats:
        cat_ = cats[centroid][0][0]
        ents = cats[centroid][1: num_topk_ents + 1]
        ents.sort(key=lambda k: k[1], reverse=True)
        ents, _ = zip(*ents)
        K.append(list(ents))
        if cat_ not in Q:
            Q[cat_] = cluster[centroid]
        else:
            Q[cat_].extend(cluster[centroid])

    QKNN = {q: {'key_words': nn, 'questions': k}
            for (q, k), nn in zip(Q.items(), K)}

    return QKNN


@plac.annotations(
    nlist=("Number of clusters", "option", "nlist", int),
    niter=("Number of iterations", "option", "niter", int),
    minlen=("Minimum length of the questions extracted",
            "option", "minlen", int),
    num_topk_ents=("Number of topk key_words", "option", "topk_ents", int),
    terminology_graph=("Build the terminology graph.",
                       "option", "term_graph", bool),
    graph_pdf_fp_out=(
        "File path where to save the PDF (default: './src/qknn_graph.gv.pdf')",
        "option", "term_graph_fp", str),
)
def main(
        papers_fp_in: Optional[str] = None,
        qknn_fp_out: Optional[str] = None,
        nlist: int = 10,
        niter: int = 20,
        minlen: int = 30,
        num_topk_ents: int = 10,
        terminology_graph: bool = False,
        graph_pdf_fp_out: Optional[str] = None,
):
    """Build the QKNN data object for the web-application."""
    config = CONFIG_DICT
    if papers_fp_in is None:
        papers_fp_in = Path(config['engine']['papers'])
        if not papers_fp_in.is_file():
            raise ValueError(f'File {papers_fp_in} does not exist. By default '
                             'the file is expected in `root/src/data/*.pkl`')

    root_path = Path(ROOT_SRC_DIR).absolute()
    if qknn_fp_out is None:
        qknn_fp_out = root_path.joinpath('QKNN.pkl')

    if graph_pdf_fp_out is None:
        graph_pdf_fp_out = root_path.joinpath('qknn_graph.gv').as_posix()

    QKNN = build_qknn(papers_filepath=papers_fp_in,
                      nlist=nlist,
                      niter=niter,
                      minlen=minlen,
                      num_topk_ents=num_topk_ents,
                      terminology_graph=terminology_graph,
                      out_graph_pdf=graph_pdf_fp_out)

    DataIO.save_data(qknn_fp_out, QKNN)
    print(f'Done! QKNN object saved in {qknn_fp_out}')
    print('Updating config.toml file with path')

    config['streamlit'].update({'qknn_file': qknn_fp_out.as_posix()})
    with open(DEFAULT_CONFIG, 'w') as f:
        toml.dump(config, f)

    print('Configuration was updated, automatically '
          'loaded when the web-app is initialized!')


if __name__ == '__main__':
    plac.call(main)
