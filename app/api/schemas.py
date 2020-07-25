from typing import List, Optional

from pydantic import BaseModel, Field


class QuestionAnsweringBase(BaseModel):
    question: str = Field(
        None,
        title="Question for the model to answer.",
        max_length=2000
    )


class QuestionAnsweringInput(QuestionAnsweringBase):
    mink: int = 15
    maxk: int = 30
    mode: str = 'bert'
    nprobe: int = 4


class QuestionAnsweringOutput(QuestionAnsweringBase):
    answer: str
    context: str
    n_sents: int
    titles: List[str]
    paper_ids: List[int]


class QuestionAnsweringWithContextInput(QuestionAnsweringBase):
    context: str


class QuestionAnsweringWithContextOutput(BaseModel):
    answer: str
    context: str


class SentenceSimilarityInput(BaseModel):
    sentence: str
    topk: Optional[int] = 5
    nprobe: Optional[int] = 1
    add_paper_ids: Optional[bool] = False


class SentenceSimilarityOutput(BaseModel):
    n_sents: int
    sents: List[str]
    dists: List[int]


class SentenceSimilarityWithPaperIdsOutput(SentenceSimilarityOutput):
    paper_ids: List[int]


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
