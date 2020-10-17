import re
from pathlib import Path
from string import punctuation
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import requests
import toml
from corona_nlp.dataset import (CORD19Dataset, clean_tokenization,
                                normalize_whitespace)

APIOutput = Union[Dict[str, Any], None]

REGX_URL = r"(((http|https)\:\/\/www\.)|((http|https)\:\/\/))|((http|https)\/{1,2})"


def app_config(toml_config: str = './config.toml') -> Dict[str, Any]:
    config_file = Path(toml_config).absolute()
    config_dict = toml.load(config_file.as_posix())
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


class MetadataReader(CORD19Dataset):
    def __init__(self, metadata_path: str, source: Union[str, List[str]]):
        super(MetadataReader, self).__init__(source)
        self.meta_df = pd.read_csv(metadata_path)
        self.paper_urls: Dict[str, str] = {}
        self._init_urls()

    def _init_urls(self):
        paper_ids = self.meta_df['sha'].tolist()
        paper_urls = None
        is_url_from_doi_key = False
        columns = list(map(lambda item: item.lower().strip(),
                           self.meta_df.columns))
        if 'url' in columns:
            paper_urls = self.meta_df['url'].to_list()
        elif 'doi' in columns:
            paper_urls = self.meta_df['doi'].to_list()
            is_url_from_doi_key = True
        else:
            raise ValueError(
                "Failed to find URLs from both possible columns: `url|doi`"
                "Make sure the Metadata CSV file has either the full url "
                "under the column `< url >` or format if `< doi >` is the "
                "column name as e.g., col:doi `10.1007/s00134-020-05985-9`"
            )
        for paper_id, url in zip(paper_ids, paper_urls):
            url = str(url).strip()
            if len(url) == 0:
                continue
            paper_id = str(paper_id).strip()
            if len(paper_id) > 0 and paper_id != 'nan' \
                    and paper_id in self.paper_index \
                    and paper_id not in self.paper_urls:

                # This url format issue has been fixed in newer versions
                # `> 2020-03-13` of the CORD-19 Dataset (kaggle version).
                # Ugly most deal with ugly. I really dont care about this.
                if is_url_from_doi_key:
                    match = re.match(REGX_URL, url)
                    if match is not None and match.group():
                        url = url.replace(match.group(), '') \
                            .replace('://', '').replace('//', '').strip()
                    url = f'https://{url}' if 'doi.org' in url \
                        else f'https://doi.org/{url}'

                self.paper_urls[paper_id] = url

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
        for index in paper_ids:
            title = self.title(index).strip()
            if len(title) == 0:
                title = 'title - n/a'
            paper_id = self[index]
            if paper_id in self.paper_urls:
                url = self.paper_urls[paper_id].strip()
                if len(url) == 0:
                    url = 'url - n/a'
                data.append({'title': title, 'url': url})
        return data
