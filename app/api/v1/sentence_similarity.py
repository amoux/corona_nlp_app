from typing import TypeVar, Union

from fastapi import APIRouter

from app.api.schemas import (SentenceSimilarityInput, SentenceSimilarityOutput,
                             SentenceSimilarityWithPaperIdsOutput)

_QAEngine = TypeVar('QAEngine')
router = APIRouter()


def search_similar(
        engine: _QAEngine,
        sentence: str,
        topk: int = 5,
        add_paper_ids: bool = False,
) -> Union[SentenceSimilarityOutput,
           SentenceSimilarityWithPaperIdsOutput]:
    """Get top-k most similar sentences from a query sentence."""
    _, ids = engine.similar(sentence, topk)
    n_sents = len(ids)
    sents = [engine.papers[sent_id] for sent_id in ids.flatten()]

    output = {'n_sents': n_sents, 'sents': sents}
    if not add_paper_ids:
        output = SentenceSimilarityOutput(**output)
    else:
        output.update({
            'paper_ids': [
                i['paper_ids'] for i in engine.papers.lookup(ids)
            ]
        })
        output = SentenceSimilarityWithPaperIdsOutput(**output)
    return output


@router.get('/sentence/', response_model=Union[
    SentenceSimilarityOutput, SentenceSimilarityWithPaperIdsOutput]
)
def sentence_similarity(s_in: SentenceSimilarityInput):
    if s_in.sentence:
        return search_similar(**s_in.dict())
