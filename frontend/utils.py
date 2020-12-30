import os
import re
from pathlib import Path
from string import punctuation
from typing import Any, Dict, List, Optional, Union, MutableMapping

import pandas as pd
import requests
import toml
from coronanlp.dataset import CORD19, clean_tokenization, normalize_whitespace

APIOutput = Union[Dict[str, Any], None]

REGX_URL = r"(((http|https)\:\/\/www\.)|((http|https)\:\/\/))|((http|https)\/{1,2})"

CORONA_APP_CONFIG = os.environ.get("CORONA_APP_CONFIG")


def app_config(toml_config: Optional[str] = None) -> MutableMapping[str, Any]:
    if toml_config is None:
        toml_config = CORONA_APP_CONFIG
    if toml_config is None:
        raise Exception(
            "CORONA_APP_CONFIG cannot be empty, make sure to "
            "export CORONA_APP_CONFIG='path/to/my_config.toml'")
    config_fp = Path(toml_config).absolute()
    if not config_fp.is_file():
        raise Exception(
            f"Expected valid TOML config file, got: {config_fp.as_posix()}")
    config_dict = toml.load(config_fp.as_posix())
    return config_dict


def count_words(string, min_word_length=2) -> int:
    tokens = clean_tokenization(normalize_whitespace(string)).split()
    words = ["".join([char for char in token if char not in punctuation])
             for token in tokens if len(token) > min_word_length]
    return len(words)


class ModelAPI:
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

    def answer(self, question: str, topk=15, top_p=30, nprobe=64, mode='bert',
               context: Optional[str] = None, port=None) -> APIOutput:
        """Return the predicted answer given the question and k-theresholds."""
        endpoint = None
        inputs = {'question': question}

        if context is not None and isinstance(context, str):
            endpoint = self.endpoints['context']
            inputs.update({'context': context})
        else:
            endpoint = self.endpoints['answer']
            inputs.update({'topk': topk, 'top_p': top_p,
                           'nprobe': nprobe, 'mode': mode})

        response = self.okay(endpoint, inputs=inputs, port=port)
        if response.status_code == 200:
            return response.json()
        return None

    def similar(
        self, text: Union[str, List[str]], top_p=5, nprobe=64, port=None,
    ) -> APIOutput:
        """Return top-k nearest sentences given a single sentence sequence.

        :param sentence: A single string sequence or a list of strings.
        """
        endpoint = self.endpoints['similar']
        inputs = {'text': text, 'top_p': top_p, 'nprobe': nprobe}
        response = self.okay(endpoint, inputs=inputs, port=port)
        if response.status_code == 200:
            return response.json()
        return None

    def tts(self, text: str, prob=0.99, port=None) -> APIOutput:
        """Return the file path of the synthesized text to audio of speech."""
        endpoint = self.endpoints['tts']
        inputs = {'text': text, 'prob': prob}
        response = self.okay(endpoint, inputs=inputs, port=port)
        if response.status_code == 200:
            return response.json()
        return None


class MetadataReader(CORD19):
    def __init__(self, metadata_path: str, source: Union[str, List[str]]):
        super(MetadataReader, self).__init__(source)
        self.meta = pd.read_csv(metadata_path)
        self.paper_urls: Dict[str, str] = {}
        self._init_urls()

    @property
    def columns(self):
        return self.meta.columns

    def _init_urls(self):
        columns = list(map(lambda i: i.lower().strip(), self.columns))
        uids = self.meta['sha'].tolist()
        urls = []
        urls_from_doi = False
        if 'url' in columns:
            urls = self.meta['url'].tolist()
        elif 'doi' in columns:
            urls = self.meta['doi'].tolist()
            urls_from_doi = True
        else:
            raise ValueError(
                "Failed to find URLs from both possible columns: `url|doi`"
                "Make sure the Metadata CSV file has either the full url "
                "under the column `< url >` or format if `< doi >` is the "
                "column name as e.g., col:doi `10.1007/s00134-020-05985-9`"
            )

        # This has to be the ugliest piece of **** code
        # - sorry to anyone looking at this but this piece
        # of **** code works, I hope to fix this one day.
        for uid, url in zip(uids, urls):
            url = str(url).strip()
            uid = str(uid).strip()
            if len(url) == 0:
                continue
            if len(uid) == 0:
                continue
            if uid == 'nan':
                continue
            if uid not in self.uid2pid:
                continue
            if uid in self.paper_urls:
                continue
            if urls_from_doi:
                m = re.match(REGX_URL, url)
                if m is not None and m.group():
                    m = m.replace(m.group(), '')
                    m = m.replace('://', '')
                    url = m.replace('//', '').strip()
                url = f'https://{url}' if 'doi.org' in url \
                    else f'https://doi.org/{url}'
            self.paper_urls[uid] = url

    def load_urls(self, output: Union[Dict[str, int], List[int]]):
        paper_ids: List[int] = None
        if isinstance(output, dict) and 'paper_ids' in output.keys():
            paper_ids = output['paper_ids']
        elif isinstance(output, list) and isinstance(output[0], int):
            paper_ids = output
        else:
            raise ValueError(
                'Expected output to be either a Dict[str, int] '
                'where the key is ``paper_ids`` and value is a '
                'list of integers or a List[int] of iterable ids'
            )
        data: Dict[str, str] = []
        for pid in paper_ids:
            title = self.title(pid).strip()
            if len(title) == 0:
                title = 'title - n/a'
            uid = self[pid]
            if uid in self.paper_urls:
                url = self.paper_urls[uid].strip()
                if len(url) == 0:
                    url = 'url - n/a'
                data.append({'title': title, 'url': url})
        return data
