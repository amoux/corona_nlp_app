from typing import List, Optional, Tuple, Union

import spacy
from spacy.lang.en import English
from spacy.tokens import Doc


def common_postags(
        doc: Union[str, List[str], Doc],
        model_name_or_nlp: Optional[Union[str, English]] = None,
) -> List[Tuple[str, int]]:
    """Obtain the top most common part-of-speech tags from a document."""
    if not isinstance(doc, Doc):
        if model_name_or_nlp is None:
            nlp = spacy.load('en_core_web_sm')
        elif isinstance(model_name_or_nlp, str):
            nlp = spacy.load(model_name_or_nlp)
        elif isinstance(model_name_or_nlp, English):
            nlp = model_name_or_nlp

        if isinstance(doc, str):
            doc = nlp(doc)
        elif isinstance(doc, list) and isinstance(doc[0], str):
            doc = nlp(' '.join(doc))

    postags = {}
    for tok in doc:
        if tok.pos_ and tok.pos_ not in postags:
            postags[tok.pos_] = 1
        else:
            postags[tok.pos_] += 1
    postags = sorted(postags.items(),
                     key=lambda k: k[1], reverse=True)

    return postags


def is_valid_paragraph(
        doc: Union[str, List[str], Doc],
        model_name_or_nlp: Optional[Union[str, English]] = None,
        invalid: Tuple[str, str] = ('PROPN', 'PUNCT', 'NUM',),
) -> bool:
    top_p = 0
    top_k = 0
    for pos, score in common_postags(doc, model_name_or_nlp):
        if pos in invalid:
            top_p += score
        else:
            top_k += score

    return False if top_p > top_k else True
