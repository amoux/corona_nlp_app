from typing import List, Union

from app.api.schemas import (QuestionAnsweringInput, QuestionAnsweringOutput,
                             QuestionAnsweringWithContextInput,
                             QuestionAnsweringWithContextOutput,
                             SentenceSimilarityInput, SentenceSimilarityOutput)
from app.api.v1.config import app_config, engine_config
from coronanlp.engine import ScibertQuestionAnswering  # type: ignore
from coronanlp.utils import load_store  # type: ignore
from fastapi import APIRouter  # type: ignore

config = app_config()
engine_kwargs = engine_config()
question_answering_kwargs = engine_kwargs.pop('question_answering')
NPROBE = question_answering_kwargs.pop('nprobe')
TOP_K = question_answering_kwargs.pop('topk')
TOP_P = question_answering_kwargs.pop('top_p')
sents = engine_kwargs.pop('sents')
index = engine_kwargs.pop('index')
is_custom_store = config['stores']['is_custom_store']

if not is_custom_store:
    sents = load_store('sents', store_name=sents)
    index = load_store('index', store_name=index)

engine = ScibertQuestionAnswering(sents, index, **engine_kwargs)
router = APIRouter()

encoder_config = engine.encoder.transformer.config
compressor_config = engine.compressor.model.config
decoder_config = engine.model.config


def engine_meta():
    devices = {k: v.type for k, v in engine.all_model_devices.items()}
    meta = {
        'sents': {
            'num_sents': engine.sents.num_sents,
            'num_papers': engine.sents.num_papers
        },
        'embed': {
            'ntotal': engine.index.ntotal,
            'd': engine.index.d
        },
        'models': {
            'sentence_encoder': {
                'device': devices['sentence_transformer_model_device'],
                'all_special_tokens': {
                    f'{t[1:-1].lower()}_token': {'id': i, 'token': t}
                    for i, t in zip(engine.tokenizer.all_special_ids,
                                    engine.tokenizer.all_special_tokens)},
                'output_hidden': encoder_config.output_hidden_states
            },
            'question_answering': {
                'device': devices['question_answering_model_device'],
                'num_parameters': engine.model.num_parameters(),
                'num_labels': engine.model.num_labels,
                'output_hidden': decoder_config.output_hidden_states
            },
            'compressors': {
                'bert_summarizer': {
                    'device': devices['summarizer_model_device'],
                    'reduce_option': engine.compressor.pooling,
                    'hidden': engine.compressor.hidden_layer,
                    'output_hidden': True  # Default since using Compressor.
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
        'devices': devices,
        'params': {
            'max_seq_len': engine.max_seq_length,
            'compressor_ratio': engine.compressor.cluster.ratio,
            'compressor_pooling': engine.compressor.pooling,
            'compressor_use_first': engine.compressor.cluster.use_first
        }
    }
    return meta


def corona_question_answering(
    question: str,
    topk: int = 5,
    top_p: int = 25,
    nprobe: int = 64,
    mode: str = 'bert',
    ratio: float = 0.2
) -> QuestionAnsweringOutput:

    topk = TOP_K if topk < 3 else topk
    top_p = TOP_P if top_p < 5 else top_p
    nprobe = NPROBE if nprobe is None or nprobe < 8 else nprobe
    kwargs = question_answering_kwargs

    engine.compressor.cluster.ratio = ratio

    pred = engine.answer(question, topk, top_p, nprobe, mode, **kwargs)

    question = pred.q
    context = pred.c
    sids = pred.sids.tolist()
    dist = pred.dist.tolist()
    pids = pred.pids.tolist()
    answers = list(map(lambda X: dict(X._asdict()), pred))

    return QuestionAnsweringOutput(
        q=question,
        c=context,
        sids=sids,
        dist=dist,
        pids=pids,
        answers=answers,
    )


def base_question_answering(
        question: str,
        context: str,
) -> QuestionAnsweringWithContextOutput:

    prediction = engine.pipeline(question=question, context=context)
    answer = prediction['answer']

    return QuestionAnsweringWithContextOutput(
        answer=answer,
        context=context,
    )


def faiss_nearest_neighbors_search(
    text: Union[str, List[str]],
    top_p: int = 5,
    nprobe: int = 64,
) -> SentenceSimilarityOutput:

    dist, sids = engine.similar(text, top_p=top_p, nprobe=nprobe)

    flat_sids = sids.squeeze(0).tolist()
    sentences = engine.sents.get(flat_sids)
    pids = [engine.sents.decode(flat_sids)]
    dist = dist.tolist()
    sids = sids.tolist()

    return SentenceSimilarityOutput(
        sids=sids,
        dist=dist,
        pids=pids,
        sentences=sentences,
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
        return corona_question_answering(
            **input_dict
        )


@router.post(
    '/question-with-context/',
    tags=["answer"],
    response_model=QuestionAnsweringWithContextOutput,
)
def question_with_context(input: QuestionAnsweringWithContextInput):
    if input.question and input.context:
        input_dict = input.dict()
        return base_question_answering(
            **input_dict
        )


@router.post(
    '/sentence-similarity/',
    tags=["similar"],
    response_model=SentenceSimilarityOutput
)
def sentence_similarity(input: SentenceSimilarityInput):
    if input.text:
        input_dict = input.dict()
        return faiss_nearest_neighbors_search(
            **input_dict
        )
