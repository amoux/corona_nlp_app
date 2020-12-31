from typing import Any, Dict, List, NamedTuple, Optional, Union

import numpy as np
import requests
from coronanlp.engine_utils import ModelOutput, QuestionAnsweringOutput

APIOutput = Union[Dict[str, Any], None]


class SentenceSimilarityOutput(NamedTuple):
    sids: np.ndarray
    dist: np.ndarray
    pids: np.ndarray
    sentences: List[str]

    @property
    def fields(self) -> List[str]:
        return list(self._fields)

    @property
    def num_sents(self) -> int:
        return self.sids.size

    def get(self, key):
        if key in self.fields:
            return self.__getattribute__(key)
        return


def parse_question_answering_response(resp) -> QuestionAnsweringOutput:
    predictions = QuestionAnsweringOutput()
    for answer in resp['answers']:
        predictions.append(ModelOutput(**answer))
    sids = np.array(resp['sids'])
    dist = np.array(resp['dist'])
    inputs = (resp['q'], resp['c'], sids, dist)
    predictions.attach_(*inputs)
    predictions.pids = np.array(resp['pids'])
    return predictions


def parse_similar_sentences_respose(resp) -> SentenceSimilarityOutput:
    sids = np.array(resp['sids'])
    dist = np.array(resp['dist'])
    pids = np.array(resp['pids'])
    sentences = resp['sentences']
    return SentenceSimilarityOutput(sids, dist, pids, sentences)


class EngineAPI:
    endpoints = {
        'answer': 'question/',
        'context': 'question-with-context/',
        'similar': 'sentence-similarity/',
        'tts': 'text-to-speech/',
        'meta': 'engine-meta/',
    }

    def __init__(self, port: Union[str, int], ip: str = "127.0.0.1"):
        self.port = str(port) if isinstance(port, int) else port
        self.url = 'http://{}:{}'.format(ip, "{port}/{endpoint}")

    def okay(self, endpoint: str, inputs=None, port=None):
        port = self.port if port is None else port
        endpoint = self.url.format(port=port, endpoint=endpoint)

        response = requests.get(endpoint) if inputs is None \
            else requests.post(endpoint, json=inputs)
        return response

    def meta(self, port=None) -> APIOutput:
        endpoint = self.endpoints['meta']

        response = self.okay(endpoint, port=port)
        if response.status_code == 200:
            return response.json()
        return None

    def answer(
        self,
        question: str,
        topk: int = 15,
        top_p: int = 30,
        nprobe: int = 64,
        mode: str = 'bert',
        context: Optional[str] = None,
        port: Optional[Union[int, str]] = None
    ) -> Optional[QuestionAnsweringOutput]:

        endpoint = None
        inputs = {'question': question}

        if context is not None and isinstance(context, str):
            endpoint = self.endpoints['context']
            inputs['context'] = context
        else:
            endpoint = self.endpoints['answer']
            inputs['topk'] = topk
            inputs['top_p'] = top_p
            inputs['nprobe'] = nprobe
            inputs['mode'] = mode

        response = self.okay(endpoint, inputs=inputs, port=port)
        if response.status_code == 200:
            return parse_question_answering_response(response.json())
        return None

    def similar(
        self, text: Union[str, List[str]], top_p=5, nprobe=64, port=None,
    ) -> Optional[SentenceSimilarityOutput]:
        """Return top-k nearest sentences given a single sentence sequence.

        :param sentence: A single string sequence or a list of strings.
        """
        endpoint = self.endpoints['similar']
        inputs = {'text': text, 'top_p': top_p, 'nprobe': nprobe}

        response = self.okay(endpoint, inputs=inputs, port=port)
        if response.status_code == 200:
            return parse_similar_sentences_respose(response.json())
        return None

    def tts(self, text: str, prob=0.99, port=None) -> APIOutput:
        """Return the file path of the synthesized text to audio of speech."""
        endpoint = self.endpoints['tts']
        inputs = {'text': text, 'prob': prob}

        response = self.okay(endpoint, inputs=inputs, port=port)
        if response.status_code == 200:
            return response.json()
        return None
