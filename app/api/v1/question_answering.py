from collections import Counter
from typing import Dict, List, Optional, Union

from corona_nlp.engine import QAEngine
from corona_nlp.utils import clean_tokenization, normalize_whitespace
from fastapi import APIRouter, Body, HTTPException

from app.api.schemas import (QuestionAnsweringBase, QuestionAnsweringInput,
                             QuestionAnsweringOutput,
                             QuestionAnsweringWithContextInput,
                             QuestionAnsweringWithContextOutput)
from app.utils import app_config

config = app_config()
API_PORT = config['fastapi']['port']
FAISS_INDEX_NPROBE = config['fastapi']['nprobe']
ENGINE_CONFIG = config['data']

router = APIRouter()
engine = QAEngine(**ENGINE_CONFIG)


def answer_question(question: str,
                    mink: int = 15,
                    maxk: int = 30,
                    mode: str = 'bert',
                    nprobe: Optional[int] = None) -> QuestionAnsweringOutput:
    """Answer the inputs and build the API data attributes."""
    if nprobe is None:
        nprobe = FAISS_INDEX_NPROBE

    output = engine.answer(question, k=mink, mode=mode, nprobe=nprobe)
    context = output['context']
    answer = output['answer'].strip()
    # If no answer then try again with maxk value with larger nprobe threshold.
    if len(answer) == 0:
        max_nprobe = max(engine.nprobe_list)
        nprobe = max_nprobe if nprobe >= max_nprobe \
            else [n for n in engine.nprobe_list if n > nprobe][0]

        output = engine.answer(question, k=maxk, mode=mode, nprobe=nprobe)
        context = output['context']
        answer = output['answer'].strip()

    sent_ids = output['ids']
    n_sents = len(sent_ids)
    paper_ids = [x['paper_ids'] for x in engine.papers.lookup(sent_ids)]
    sorted_top_k_paper_ids, _ = zip(*Counter(paper_ids).most_common())
    titles = list(engine.titles(sorted_top_k_paper_ids))

    return QuestionAnsweringOutput(
        question=question,
        answer=answer,
        context=context,
        n_sents=n_sents,
        titles=titles,
        paper_ids=sorted_top_k_paper_ids,
    )


def answer_question_with_context(
        question: str, context: str) -> QuestionAnsweringWithContextOutput:
    answer, context = engine.decode(question, context)
    return QuestionAnsweringWithContextOutput(answer=answer, context=context)


@router.get('/question/', response_model=QuestionAnsweringOutput)
async def question(q_in: QuestionAnsweringInput):
    input_dict = q_in.dict()
    if input_dict.question:
        return answer_question(**input_dict.dict())


@router.get(
    '/question-with-context/',
    response_model=QuestionAnsweringWithContextOutput,
)
async def question_with_context(q_in: QuestionAnsweringWithContextInput):
    if q_in.question and q_in.context:
        return answer_question_with_context(**q_in.dict())
