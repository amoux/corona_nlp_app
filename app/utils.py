import re
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd
import requests
import spacy
import toml
from corona_nlp.dataset import CORD19Dataset
from spacy import displacy


def app_config(toml_config: str = 'config.toml') -> Dict[str, Any]:
    from app import __path__ as app_path
    if not isinstance(app_path, list):
        if hasattr(app_path, "_path"):
            app_path = app_path._path[0]
    else:
        app_path = app_path[0]
    app_home = Path(app_path).parent
    config_file = app_home.joinpath(toml_config)
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
    url = "http://{}:{}/api/v1/corona_nlp/{}"

    def __init__(self, port: Union[str, int], ip: str = "127.0.0.1"):
        if isinstance(port, int):
            port = str(port)
        self.port = port
        self.url = self.url.format(ip, "{port}", "{endpoint}",)

    def question_answering(
            self, question: str, context: str = None,
            min_k=25, max_k=100, mode="spacy", port=None) -> Dict[str, Any]:
        """Return the predicted answer given the question and k-theresholds."""
        if port is None:
            port = self.port

        server = self.url.format(port=port, endpoint="answer")
        data = f"{server}?question={question}"
        if context is not None:
            data = data + f"&context={context}"

        data = data + f"&min_k={min_k}&max_k={max_k}&mode={mode.lower()}"
        resp = requests.get(data)
        if resp.status_code == 200:
            return resp.json()

    def sentence_similarity(self, sentence: str, top_k=5,
                            with_paper_ids=True, port=None) -> Dict[str, Any]:
        """Return top-k nearest sentences given a single sentence sequence."""
        if port is None:
            port = self.port

        server = self.url.format(port=port, endpoint="similar")
        data = f"{server}?sequence={sentence}&top_k={top_k}"
        if with_paper_ids:
            data = data + "&paper_ids"

        resp = requests.get(data)
        if resp.status_code == 200:
            return resp.json()

    def text_to_speech(
            self, context: str, k=0.99, port=None) -> Dict[str, str]:
        """Return the synthesized (preprocessed) context to audio file path."""
        if port is None:
            port = self.port

        server = self.url.format(port=port, endpoint="audio_file")
        data = f"{server}?text={context}&k={k}"
        resp = requests.get(data)
        if resp.status_code == 200:
            return resp.json()


class MetadataReader(CORD19Dataset):
    def __init__(self, metadata_path: str, source: Union[str, List[str]]):
        super(MetadataReader, self).__init__(source)
        self.meta_df = pd.read_csv(metadata_path)
        self.paper_urls: Dict[str, str] = {}
        self._init_urls()

    def _init_urls(self):
        paper_ids, urls = (self.meta_df['sha'].to_list(),
                           self.meta_df['url'].to_list())
        for id, url in zip(paper_ids, urls):
            id = str(id).strip()
            if (
                len(id) > 0
                and id != 'nan'
                and id in self.paper_index
                and id not in self.paper_urls
            ):
                self.paper_urls[id] = url

    def load_urls(
            self, output: Union[Dict[str, int], List[int]]) -> Dict[str, Any]:
        paper_ids = None
        if isinstance(output, dict) and 'paper_ids' in output:
            paper_ids = output['paper_ids']
        elif isinstance(output, list) and isinstance(output[0], int):
            paper_ids = output
        else:
            raise ValueError('Expected output to be either a Dict[str, int] '
                             'where the key is ``paper_ids`` and value is a '
                             'list of integers or a List[int] of iterable ids')

        data = []
        for id in output['paper_ids']:
            title = self.title(id).strip()
            if len(title) == 0:
                title = 'title - n/a'

            paper_id = self[id]
            if paper_id in self.paper_urls:
                url = self.paper_urls[paper_id].strip()
                if len(url) == 0:
                    url = 'url - n/a'

                data.append({'title': title, 'url': url})

        return data
