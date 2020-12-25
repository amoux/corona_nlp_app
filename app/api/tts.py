import json
from pathlib import Path
from typing import IO, Dict, List, Optional, Union

import ibm_cloud_sdk_core.authenticators as ibm_auth  # type: ignore
import ibm_cloud_sdk_core.detailed_response as ibm_resp  # type: ignore
import spacy  # type: ignore
from ibm_watson import TextToSpeechV1  # type: ignore
from pydub import AudioSegment, playback  # type: ignore
from spacy.language import Language  # type: ignore

from .utils import is_valid_paragraph


class IBMTextToSpeech:
    def __init__(
            self,
            apikey: str,
            url: str,
            meta_id: int = 0,
            load_meta_files: bool = True,
            cache_dir: str = "tts-data",
            auto_save: bool = True,
            nlp_model: str = "en_core_web_md",
            spacy_nlp: Language = None,
            disable_ssl: bool = False,
            voice: str = "en-US_MichaelV3Voice",
            audio_format: str = "audio/wav",
    ) -> None:
        """Watson text to speech utility class.

        :param apikey: Watson text-to-speech IAM-API key.
        :param url: A Watson URL related to the API key.
        :param meta_id: ID of the meta config to load if exists.
        :param load_meta_files: True loads an existsting source of audio files.
        :param cache_dir: Directory where all the files will be saved.
        :param disable_dir: Disable SSL for all requests to WatsonCloud.
        """
        self._session_audio_files: List[int] = []
        self._session_audio_index = 0
        self._cachedir: Optional[Union[str, Path]] = None
        self._metafile: Optional[Union[str, Path]] = None
        self.has_metafile = False
        self.audio_file_suffix = audio_format.split('/')[-1]
        self.queries: Dict[int, Dict[str, str]] = {}
        self.cache_dir = Path(cache_dir)
        self.auto_save = auto_save
        self.audio_format = audio_format
        self.voice = voice
        self.text_to_speech = TextToSpeechV1(ibm_auth.IAMAuthenticator(apikey))
        self.text_to_speech.set_service_url(url)
        self.disable_ssl() if disable_ssl else self.enable_ssl()
        self.nlp = spacy_nlp if isinstance(spacy_nlp, Language) \
            else spacy.load(nlp_model)
        if meta_id != -1:
            self.load_meta(meta_id, load_meta_files=load_meta_files)

    def disable_ssl(self) -> None:
        self.text_to_speech.set_disable_ssl_verification(True)

    def enable_ssl(self) -> None:
        self.text_to_speech.set_disable_ssl_verification(False)

    def is_paragraph_valid(self, sequence: str) -> bool:
        return is_valid_paragraph(sequence, model_name_or_nlp=self.nlp)

    def list_voices(self) -> List[Dict[str, str]]:
        return self.text_to_speech.list_voices().get_result()["voices"]

    def delete_query(self, index: int, del_audio_file=True, update_meta=True):
        """Properly delete an existing query + audio file in a session.

        This method should be used whenever a query or audio file is
        removed. Since the meta file and instance index are updated
        to handle the changes accordingly.
        """
        assert index in self.queries, f'Query index[{index}] not found.'
        assert isinstance(self._cachedir, Path)
        file = self._cachedir.joinpath(self.queries[index]['file'])
        maxid = max(self.queries.keys())
        if index == maxid:
            self.queries.pop(index)
            if index == self._session_audio_index - 1:
                self._session_audio_index = maxid
        elif index < maxid and index not in self._session_audio_files:
            self.queries[index].clear()
            self._session_audio_files.append(index)
        if del_audio_file and file.is_file():
            file.unlink()
        if update_meta:
            self.save_meta()

    def save_meta(self, meta_file: Optional[Union[str, Path]] = None) -> None:
        if isinstance(meta_file, str):
            meta_file = Path(meta_file)
        elif meta_file is None:
            meta_file = self._metafile
        assert isinstance(meta_file, Path)

        with meta_file.open('w') as file:
            data = json.dumps(self.queries, indent=4, separators=(",", ": "))
            file.write(data)

    def load_meta(self, meta_id=0, inplace=True, cache_dir: Path = None,
                  suffix: str = None, load_meta_files=False):
        """Load the meta data containing the query texts and audio files."""
        if suffix is None:
            suffix = f'*.{self.audio_file_suffix}'
        if cache_dir is None:
            cache_dir = self.cache_dir
        if not cache_dir.is_dir():
            cache_dir.mkdir(parents=True)
        if meta_id != 0 and not load_meta_files:
            load_meta_files = True

        meta_file = cache_dir.joinpath(f'meta_{meta_id}.json')
        files_dir = cache_dir.joinpath(f'audio_{meta_id}')

        if load_meta_files:
            if files_dir.is_dir() and meta_file.is_file():
                files = [f for f in files_dir.glob(suffix) if f.is_file()]
                if len(files) == 0:
                    raise ValueError('Meta file exists but found directory:'
                                     f' {files_dir} with no audio files.')
            elif meta_id == 0:
                pass

        elif not load_meta_files and files_dir.exists():
            audio_bins = []
            for item in cache_dir.iterdir():
                if item.is_dir():
                    nfiles = [f for f in item.glob(suffix) if f.is_file()]
                    dir_id = item.name.split('_')[-1]
                    if dir_id.isdigit():
                        audio_bins.append((int(dir_id), len(nfiles)))

            audio_bins = sorted(audio_bins, key=lambda k: k[1])
            maxid = max(audio_bins)[0] + 1
            if audio_bins[0][1] == 0:
                maxid = audio_bins[0][0]
            meta_file = cache_dir.joinpath(f'meta_{maxid}.json')
            files_dir = cache_dir.joinpath(f'audio_{maxid}')
            setattr(self, 'audio_bins', audio_bins)

        if not files_dir.is_dir():
            files_dir.mkdir()
        if meta_file.is_file():
            queries = {}
            index = 0
            with meta_file.open("r") as file:
                metadata = json.load(file)
                for i, data in metadata.items():
                    queries[int(i)] = data
                    index = max(int(i), index)

            if inplace:
                self._session_audio_index = index + 1
                self.queries = queries
                self.has_metafile = True
            else:
                return queries

        self._cachedir = files_dir
        self._metafile = meta_file

    def is_similar(self, text: str, to: str, k=0.99) -> bool:
        """Evaluate how similar an existing text is to another.

        :param k: How similar A to B need to be to be considered as similar.
            The greater the value - the more similar both possible
            matches need to be in order to be considered as similar.
        """
        score = self.nlp(text).similarity(self.nlp(to))
        return True if score >= k else False

    def smart_cache(self, text: str, k=0.99) -> int:
        """Return the file index if the text is similar to a cached text."""
        similar = -1
        for index, data in self.queries.items():
            if self.is_similar(data["text"], to=text, k=k):
                similar = index
                break
        return similar

    def synthesize(self, text: str, audio_format: Optional[str] = None
                   ) -> ibm_resp.DetailedResponse:
        if audio_format is None:
            audio_format = self.audio_format
        return self.text_to_speech.synthesize(
            text=text,
            voice=self.voice,
            accept=self.audio_format
        )

    def play_synth(self, file: Path, suffix: str = None) -> None:
        sound_file = self.load_synth(file=file, suffix=suffix)
        playback.play(sound_file)

    def load_synth(self, file: Path, suffix: str = None) -> AudioSegment:
        """Load an audio file from file and return its path."""
        if suffix is None:
            suffix = self.audio_file_suffix
        if not isinstance(file, Path):
            file_a = self.cache_dir.joinpath(file)
            file_b = self._cachedir.joinpath(file)
            file = file_a if file_a.is_file() else file_b
        sound_file = AudioSegment.from_file(file, format=suffix)
        return sound_file

    def write_synth(self, file: Path, text: str) -> None:
        if file.exists() and not self.has_metafile:
            raise ValueError('Warning, file exists and '
                             'metadata has not been loaded.')
        with file.open("wb") as audio_file:
            result = self.synthesize(text).get_result()
            audio_file.write(result.content)

    def encode_tts(self, text: str, k=0.99, play_file=True, suffix=None):
        assert isinstance(self._cachedir, Path)
        if suffix is None:
            suffix = self.audio_file_suffix
        file = None
        text = text.strip().lower()
        index = self.smart_cache(text, k=k)
        if isinstance(index, int):
            name = self.queries[index]['file']
            file = self._cachedir.joinpath(name)
        else:
            if len(self._session_audio_files) > 0:
                index = self._session_audio_files.pop(0)
            else:
                index = self._session_audio_index
                self._session_audio_index += 1
            name = f'{index}_speech.{suffix}'
            file = self._cachedir.joinpath(name)
            self.queries[index] = {'file': name, 'text': text}
            self.write_synth(file, text)
            if self.auto_save:
                self.save_meta()
        if file is not None:  # play existing or cached file.
            if play_file:
                self.play_synth(file, suffix=suffix)
            else:
                return file

    def __getitem__(self, item):
        return self.queries[item]

    def __call__(self, text: str, k=0.99, play_file=True):
        """Encode text to speech and play or return the path of the audio file.

        Returns the path to the audio file if play_file is set to False.
            Otherwise, the synthesized audio file is played directly.
        """
        if play_file:
            self.encode_tts(text, k)
        else:
            return self.encode_tts(text, k, play_file=False)
