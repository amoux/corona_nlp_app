from collections import Counter
from typing import Iterable, List, Optional, Tuple, Union

import spacy  # type: ignore
from spacy.language import Language  # type: ignore
from spacy.pipeline import Tagger  # type: ignore
from spacy.tokens import Doc  # type: ignore


def part_of_speech(texts: Iterable[str], nlp: Optional[Language] = None,
                   model: str = 'en_core_web_sm') -> Counter:
    """Part-Of-Speech Tags from an Iterable of String Sequences."""
    added_tagger: Union[str, bool] = False
    if nlp is None:
        nlp = spacy.load(model)
    elif isinstance(nlp, Language) and 'tagger' not in nlp.pipe_names:
        nlp.add_pipe(Tagger(nlp.vocab))
        added_tagger = 'tagger'

    pos_tags = {}
    for text in texts:
        if not text:
            continue
        for tok in nlp(text):
            if tok.is_space or not tok.is_alpha or not tok.pos_:
                continue
            if tok.pos_ not in pos_tags:
                pos_tags[tok.pos_] = 1
            else:
                pos_tags[tok.pos_] += 1

    if added_tagger:
        nlp.remove_pipe(added_tagger)
    return Counter(pos_tags)


def common_postags(doc, model_name_or_nlp) -> List[Tuple[str, int]]:
    """Part of Speech Tags from an iterable of string sequences.

    :param doc: (Iterable[str]) It can be any iterable of strings.
    :param model_name_or_nlp: (str) Language instance or the name
        of the model to use.

    NOTE: This method will be deleted. Use `part_of_speech()` instead.
    """
    kwargs = dict(texts=doc)
    if isinstance(model_name_or_nlp, str):
        kwargs.update({'model': model_name_or_nlp})
    elif isinstance(model_name_or_nlp, Language):
        kwargs.update({'nlp': model_name_or_nlp})

    postags = part_of_speech(**kwargs)
    common = list(postags.items())
    return common


def is_valid_paragraph(
        doc: Union[str, List[str], Doc],
        model_name_or_nlp: Optional[Union[str, Language]] = None,
        invalid_postags: Tuple[str, ...] = ('PROPN', 'PUNCT', 'NUM',)
) -> bool:
    top_p, top_k = 0, 0
    for pos, score in common_postags(doc, model_name_or_nlp):
        if pos in invalid_postags:
            top_p += score
        else:
            top_k += score
    return False if top_p > top_k else True
