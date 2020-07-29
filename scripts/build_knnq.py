from pathlib import Path
from typing import Optional

import plac
import toml
from corona_nlp.datatypes import Papers
from corona_nlp.retrival import extract_questions
from corona_nlp.transformer import SentenceTransformer
from corona_nlp.utils import DataIO

from .utils import CONFIG_DICT


def main(config_file: Optional[str] = None, minlen=40, nlist=10):
    config = None
    if config_file is None:
        config = CONFIG_DICT
    else:
        config = toml.load(Path(config_file).absolute().as_posix())
        if not isinstance(config, dict):
            raise ValueError(f'Could not load config file from: {config_file}')

    # get all the questions from the preprocessed sentences
    papers = Papers.from_disk(config['engine']['papers'])
    questions = extract_questions(papers, minlen, sentence_ids=False)

    # load the encoder to encode questions to embeddings
    encoder = SentenceTransformer(config['models']['sentence_transformer'])
    embedding = encoder.encode(questions)

    topk = len(questions) // nlist
