from collections import Counter
from typing import Any, Dict, List, Optional, Union

from corona_nlp.engine import QAEngine
from corona_nlp.utils import clean_tokenization, normalize_whitespace
from fastapi import APIRouter, Body, HTTPException

from app.api.schemas import (QuestionAnsweringInput, QuestionAnsweringOutput,
                             QuestionAnsweringWithContextInput,
                             QuestionAnsweringWithContextOutput,
                             SentenceSimilarityInput, SentenceSimilarityOutput,
                             SentenceSimilarityWithPaperIdsOutput)
from app.utils import app_config

config = app_config()
ENGINE_CONFIG = config['engine']
ENGINE_CONFIG.update({'source': config['cord']['source']})
ENGINE_CONFIG.update({
    'encoder': config['models']['sentence_transformer'],
    'model': config['models']['question_answering'],
    'nlp_model': config['models']['spacy_nlp']
})
ENGINE_CONFIG.update(config['cord']['init'])
FAISS_INDEX_NPROBE = config['fastapi']['nprobe']

router = APIRouter()
engine = QAEngine(**ENGINE_CONFIG)


def engine_meta() -> Dict[str, Dict[str, Union[str, int, Dict[str, Any]]]]:
    devices = {k: v.type for k, v in engine.engine_devices.items()}
    meta = {
        'string_store': {
            'num_sents': engine.papers.num_sents,
            'num_papers': engine.papers.num_papers
        },
        'embedding_store': {
            'ntotal': engine.index.ntotal,
            'd': engine.index.d
        },
        'models': {
            'sentence_transformer': {
                'model_name_or_path': ENGINE_CONFIG['encoder'],
                'device': devices['sentence_transformer_model_device'],
                'max_seq_length': engine.encoder.max_seq_length,
                'all_special_tokens': {
                    f'{t[1:-1].lower()}_token': {'id': i, 'token': t}
                    for i, t in zip(engine.tokenizer.all_special_ids,
                                    engine.tokenizer.all_special_tokens)
                }
            },
            'question_answering': {
                'model_name_or_path': ENGINE_CONFIG['model'],
                'device': devices['question_answering_model_device'],
                'num_parameters': engine.model.num_parameters(),
                'num_labels': engine.model.num_labels
            },
            'compressors': {
                'bert_summarizer': {
                    'device': devices['summarizer_model_device'],
                    'reduce_option': engine._bert_summarizer.reduce_option,
                    'hidden': engine._bert_summarizer.hidden,
                    'custom_model': ENGINE_CONFIG['model'],
                    'custom_tokenizer': ENGINE_CONFIG['encoder']
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


def answer(question: str,
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

    if len(answer) == 0:
        max_nprobe = max(engine.nprobe_list)
        nprobe = max_nprobe if nprobe >= max_nprobe \
            else [n for n in engine.nprobe_list if n > nprobe][0]

        output = engine.answer(question, k=maxk, mode=mode, nprobe=nprobe)
        context = output['context']
        answer = output['answer'].strip()

    sent_ids = output['ids']
    n_sents = len(sent_ids)
    paper_ids = [x['paper_id'] for x in engine.papers.lookup(sent_ids)]
    top_k_paper_ids, _ = zip(*Counter(paper_ids).most_common())
    top_k_paper_ids = list(top_k_paper_ids)
    titles = list(engine.titles(top_k_paper_ids))

    return QuestionAnsweringOutput(
        question=question,
        answer=answer,
        context=context,
        n_sents=n_sents,
        titles=titles,
        paper_ids=top_k_paper_ids,
    )


def decode(question: str, context: str) -> QuestionAnsweringWithContextOutput:
    answer, context = engine.decode(question, context)
    return QuestionAnsweringWithContextOutput(answer=answer, context=context)


def similar(sentence: str,
            topk: int = 5,
            nprobe: int = 1,
            add_paper_ids: bool = False) -> Union[
                SentenceSimilarityOutput,
                SentenceSimilarityWithPaperIdsOutput]:
    dists, indices = engine.similar(sentence, k=topk, nprobe=nprobe)
    sentences = [engine.papers[id] for id in indices.flatten()]
    output = {
        'n_sents': len(sentences),
        "sents": sentences,
        "dists": dists.tolist()[0]
    }
    if not add_paper_ids:
        output = SentenceSimilarityOutput(**output)
    else:
        output.update({
            'paper_ids': [
                i['paper_id'] for i in engine.papers.lookup(indices.flatten())
            ]
        })
        output = SentenceSimilarityWithPaperIdsOutput(**output)
    return output


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
    response_model=Union[SentenceSimilarityOutput,
                         SentenceSimilarityWithPaperIdsOutput]
)
def sentence_similarity(input: SentenceSimilarityInput):
    if input.sentence:
        input_dict = input.dict()
        return similar(**input_dict)
