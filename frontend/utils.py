import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import requests
import spacy
import toml
from corona_nlp.dataset import CORD19Dataset
from spacy import displacy


def app_config(toml_config: str = './config.toml') -> Dict[str, Any]:
    config_file = Path(toml_config).absolute()
    config_dict = toml.load(config_file.as_posix())
    return config_dict


def render(question: str, prediction: Dict[str, str], jupyter=True,
           return_html=False, style="ent", manual=True, label='ANSWER'):
    """Spacy displaCy visualization util for the question answering model."""
    options = {"compact": True, "bg": "#ed7118", "color": '#000000'}
    display_data = {}
    start, end = 0, 0
    match = re.search(prediction["answer"], prediction["context"])
    if match and match.span() is not None:
        start, end = match.span()

    display_data["ents"] = [{'start': start, 'end': end, 'label': label}]
    options['ents'] = [label]
    options['colors'] = {label: "linear-gradient(90deg, #aa9cfc, #fc9ce7)"}
    if len(prediction['context']) > 1:
        display_data['text'] = prediction['context']

    display_data['title'] = f'Q : {question}\n'
    if return_html:
        return displacy.render([display_data], style=style,
                               jupyter=False, options=options, manual=manual)
    else:
        displacy.render([display_data], style=style,
                        page=False, minify=True,
                        jupyter=jupyter, options=options, manual=manual)


class ModelAPI:
    url = "http://{}:{}/{}"

    def __init__(self, port: Union[str, int], ip: str = "127.0.0.1"):
        if isinstance(port, int):
            port = str(port)
        self.port = port
        self.url = self.url.format(ip, "{port}", "{endpoint}",)

    def engine_meta(self, port: Optional[int] = None) -> Dict[str, Tuple[Any, ...]]:
        request_url = self.url.format(
            port=port if port is not None else self.port,
            endpoint='engine-meta/'
        )
        response = requests.get(request_url)
        if response.status_code == 200:
            return response.json()

    def question_answering(self,
                           question: str,
                           context: Optional[str] = None,
                           mink: int = 15,
                           maxk: int = 30,
                           mode: str = "bert",
                           nprobe: int = 1,
                           port: Optional[int] = None) -> Dict[str, Any]:
        """Return the predicted answer given the question and k-theresholds."""
        input_dict, endpoint = {"question": question}, "question/"
        if context is not None and isinstance(context, str):
            endpoint = "question-with-context/"
            input_dict.update({"context": context})
        else:
            input_dict.update({
                "mink": mink,
                "maxk": maxk,
                "mode": mode,
                "nprobe": nprobe
            })
        request_url = self.url.format(
            port=port if port is not None else self.port,
            endpoint=endpoint
        )
        response = requests.post(request_url, json=input_dict)
        if response.status_code == 200:
            return response.json()

    def sentence_similarity(self,
                            sentence: str,
                            topk: int = 5,
                            nprobe: int = 1,
                            port: Optional[int] = None) -> Dict[str, Any]:
        """Return top-k nearest sentences given a single sentence sequence."""
        request_url = self.url.format(
            port=port if port is not None else self.port,
            endpoint="sentence-similarity/"
        )
        input_dict = {
            'sentence': sentence,
            'topk': topk,
            'nprobe': nprobe
        }
        response = requests.post(request_url, json=input_dict)
        if response.status_code == 200:
            return response.json()

    def text_to_speech(self,
                       text: str,
                       prob: float = 0.99,
                       port: Optional[int] = None) -> Dict[str, str]:
        """Return the file path of the synthesized text to audio of speech."""
        request_url = self.url.format(
            port=port if port is not None else self.port,
            endpoint='text-to-speech/'
        )
        input_dict = {'text': text, 'prob': prob}
        response = requests.post(request_url, json=input_dict)
        if response.status_code == 200:
            return response.json()


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
        if 'url' in self.meta_df.columns:
            paper_urls = self.meta_df['url'].to_list()
        elif 'doi' in self.meta_df.columns:
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
            paper_id = str(paper_id).strip()
            if len(paper_id) > 0 and paper_id != 'nan' \
                    and paper_id in self.paper_index \
                    and paper_id not in self.paper_urls:
                if is_url_from_doi_key:
                    # This url format issue has been fixed in newer versions
                    # `> 2020-03-13` of the CORD-19 Dataset (kaggle version).
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
