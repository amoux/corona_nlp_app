from typing import Optional, Union

from app.api.schemas import TextToSpeechInput, TextToSpeechOutput
from app.api.tts import IBMTextToSpeech
from app.utils import app_config
from coronanlp.utils import (clean_tokenization,  # type: ignore
                             normalize_whitespace)
from fastapi import APIRouter  # type: ignore

config = app_config()

kwargs = {}
kwargs['apikey'] = config['tts']['init']['apikey']
kwargs['disable_ssl'] = config['tts']['init']['disable_ssl']
kwargs['cache_dir'] = config['tts']['cache_dir']
kwargs['voice'] = config['tts']['voice']['name']
default_url = config['tts']['init']['default_url']
kwargs['url'] = config['tts']['init'][default_url]

router = APIRouter()
ibm_tts = IBMTextToSpeech(**kwargs)


def synthesize(
    text: str,
    prob: float = 0.99
) -> Union[TextToSpeechOutput, None]:

    tts_output: Optional[TextToSpeechOutput] = None
    text = normalize_whitespace(text)
    text = clean_tokenization(text)
    audio_file = None
    if ibm_tts.is_paragraph_valid(text):
        audio_file = ibm_tts(text, k=prob, play_file=False, websocket=True)
    if audio_file is not None:
        audio_file_path = audio_file.absolute().as_posix()
        tts_output = TextToSpeechOutput(audio_file_path=audio_file_path)
    return tts_output


@router.post(
    '/text-to-speech/',
    tags=['audio_fle'],
    response_model=TextToSpeechOutput
)
async def text_to_speech(input: TextToSpeechInput):
    if input.text:
        input_dict = input.dict()
        return synthesize(**input_dict)
