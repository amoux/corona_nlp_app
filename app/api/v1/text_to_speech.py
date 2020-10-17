from typing import Optional, Union
from corona_nlp.utils import clean_tokenization, normalize_whitespace
from fastapi import APIRouter

from app.api.schemas import TextToSpeechInput, TextToSpeechOutput
from app.api.tts import IBMTextToSpeech
from app.utils import app_config

config = app_config()
TTS_CONFIG = config['tts']['init']
TTS_CONFIG.update({'cache_dir': config['tts']['cache_dir']})
TTS_CONFIG.update({'voice': config['tts']['voice']['name']})

router = APIRouter()
ibm_tts = IBMTextToSpeech(**TTS_CONFIG)


def synthesize(text: str, prob: float = 0.99
               ) -> Union[TextToSpeechOutput, None]:
    tts_output: Optional[TextToSpeechOutput] = None
    text = clean_tokenization(normalize_whitespace(text))

    if ibm_tts.is_paragraph_valid(sequence=text):
        audio_file = ibm_tts(text=text, k=prob, play_file=False)
        if audio_file is not None:
            audio_file = audio_file.absolute().as_posix()
            tts_output = TextToSpeechOutput(audio_file_path=audio_file)

    return tts_output


@router.post(
    '/text-to-speech/',
    tags=['audio_fle'],
    response_model=TextToSpeechOutput
)
def text_to_speech(input: TextToSpeechInput):
    if input.text:
        input_dict = input.dict()
        return synthesize(**input_dict)
