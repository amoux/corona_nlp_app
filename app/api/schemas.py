from typing import List, Optional, Union, Tuple, Dict, Any

from pydantic import BaseModel, Field


class QuestionAnsweringBase(BaseModel):
    question: str = Field(
        None,
        title="Question for the model to answer.",
        max_length=2000
    )


class QuestionAnsweringInput(QuestionAnsweringBase):
    topk: int = 5
    top_p: int = 25
    mode: str = 'bert'
    nprobe: int = 64


class QuestionAnsweringOutput(BaseModel):
    q: Union[str, List[str]]
    c: Union[str, List[str]]
    sids: List[List[int]]
    dist: List[List[float]]
    pids: List[List[int]]
    answers: List[Dict[str, Any]]


class QuestionAnsweringWithContextInput(QuestionAnsweringBase):
    context: str


class QuestionAnsweringWithContextOutput(BaseModel):
    answer: str
    context: str


class SentenceSimilarityInput(BaseModel):
    text: Union[str, List[str]]
    top_p: Optional[int] = 5
    nprobe: Optional[int] = 1


class SentenceSimilarityOutput(BaseModel):
    sids: List[List[int]]
    dist: List[List[float]]
    pids: List[List[int]]
    sentences: List[str]


class TextToSpeechInput(BaseModel):
    text: str
    prob: float = Field(0.99, description=(
        'Probability/similarity threshold for comparing a new text input to an'
        'existing text input e.g., if text\'s A and B have a similarity score '
        'of `0.99` then the probability of both A & B are the same is ~ HIGH| '
        'Which means the audio file of B is loaded instead of calling the API '
        'to encode A (the new text input) to audio. This technique avoids the '
        'unnecessary calls and reduces latency since the audio is loaded from '
        'disk directly.')
    )


class TextToSpeechOutput(BaseModel):
    audio_file_path: str = Field(
        None,
        description='Path to the encoded audio file.'
    )
