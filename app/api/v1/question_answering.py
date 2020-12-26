from typing import Any, Dict, List, Union

from app.api.schemas import (QuestionAnsweringInput, QuestionAnsweringOutput,
                             QuestionAnsweringWithContextInput,
                             QuestionAnsweringWithContextOutput,
                             SentenceSimilarityInput, SentenceSimilarityOutput)
from app.api.v1.config import app_config, engine_config
from coronanlp.engine import ScibertQuestionAnswering
from coronanlp.utils import load_store
from fastapi import APIRouter  # type: ignore


config = app_config('config.toml')
is_custom_store = config['store']['is_custom_store']

engine_kwargs = engine_config('config.toml')
sents = engine_kwargs.pop('sents')
index = engine_kwargs.pop('index')
if not is_custom_store:
    sents = load_store('sents', store_name=sents)
    index = load_store('index', store_name=index)

engine = ScibertQuestionAnswering(sents, index, **engine_kwargs)
router = APIRouter()

FAISS_INDEX_NPROBE = config['fastapi']['nprobe']
ENGINE_KWARGS = engine_kwargs


def engine_meta() -> Dict[str, Dict[str, Union[str, int, Dict[str, Any]]]]:
    devices = {k: v.type for k, v in engine.all_model_devices.items()}
    meta = {
        'string_store': {
            'num_sents': engine.sents.num_sents,
            'num_papers': engine.sents.num_papers
        },
        'embedding_store': {
            'ntotal': engine.index.ntotal,
            'd': engine.index.d
        },
        'models': {
            'encoder': {
                'model_name_or_path': ENGINE_KWARGS['encoder'],
                'device': devices['encoder_model_device'],
                'max_seq_length': engine.encoder.max_length,
                'all_special_tokens': {
                    f'{t[1:-1].lower()}_token': {'id': i, 'token': t}
                    for i, t in zip(engine.tokenizer.all_special_ids,
                                    engine.tokenizer.all_special_tokens)
                }
            },
            'question_answering': {
                'model_name_or_path': ENGINE_KWARGS['model'],
                'device': devices['question_answering_model_device'],
                'num_parameters': engine.model.num_parameters(),
                'num_labels': engine.model.num_labels
            },
            'compressors': {
                'bert_summarizer': {
                    'device': devices['summarizer_model_device'],
                    'reduce_option': engine._bert_summarizer.reduce_option,
                    'hidden': engine._bert_summarizer.hidden,
                    'custom_model': ENGINE_KWARGS['model'],
                    'custom_tokenizer': ENGINE_KWARGS['encoder']
                },
                'freq_summarizer': {
                    'lang': engine.nlp.meta['lang'],
                    'name': engine.nlp.meta['name'],
                    'spacy_version': engine.nlp.meta['spacy_version'],
                    'speed': engine.nlp.meta['speed'],
                    'description': engine.nlp.meta['description']
                }
            }
        },
        'devices': devices
    }
    return meta


def answer(question: str, topk: int = 5, top_p: int = 25, nprobe: int = 64,
           mode: str = 'bert') -> QuestionAnsweringOutput:
    """Answer the inputs and build the API data attributes."""
    if nprobe is None:
        nprobe = FAISS_INDEX_NPROBE

    pred = engine.answer(question, topk, top_p, nprobe, mode=mode)
    pred.popempty()
    sids = pred.sids.unsqueeze(0).tolist()
    pids = list(engine.sents.lookup(sids, mode='table').keys())
    titles = list(engine.cord19.titles(pids))
    num_sents = len(sids)

    # TODO: The engine is now able to provide multiple answers! The previous
    # engine could only provide a single answer. Which means I need to update.
    # the methods that use the inputs or/and outputs by this method.

    answer = pred[0].answer
    context = pred.context

    return QuestionAnsweringOutput(
        question=question,
        answer=answer,
        context=context,
        num_sents=num_sents,
        titles=titles,
        paper_ids=pids,
    )


def decode(question: str, context: str) -> QuestionAnsweringWithContextOutput:
    prediction = engine.pipeline(question=question, context=context)
    answer = prediction['answer']
    return QuestionAnsweringWithContextOutput(answer=answer, context=context)


def similar(text: Union[str, List[str]], top_p: int = 5, nprobe: int = 64,
            ) -> SentenceSimilarityOutput:
    dist, sids = engine.similar(text, top_p=top_p, nprobe=nprobe)
    dist = dist.squeeze(0).tolist()
    sids = sids.squeeze(0).tolist()
    pids = engine.decode(sids)
    sentences = engine.get(sids)
    num_sents = len(sentences)

    return SentenceSimilarityOutput(
        num_sents=num_sents,
        sents=sentences,
        dists=dist,
        paper_ids=pids,
    )


@router.get('/engine-meta/', tags=['meta'])
def get_engine_meta():
    return engine_meta()


@router.post(
    '/question/',
    tags=["answer"],
    response_model=QuestionAnsweringOutput
)
def question(input: QuestionAnsweringInput):
    if input.question:
        input_dict = input.dict()
        return answer(**input_dict)


@router.post(
    '/question-with-context/',
    tags=["answer"],
    response_model=QuestionAnsweringWithContextOutput,
)
def question_with_context(input: QuestionAnsweringWithContextInput):
    if input.question and input.context:
        input_dict = input.dict()
        return decode(**input_dict)


@router.post(
    '/sentence-similarity/',
    tags=["similar"],
    response_model=SentenceSimilarityOutput
)
def sentence_similarity(input: SentenceSimilarityInput):
    if input.text:
        input_dict = input.dict()
        return similar(**input_dict)
