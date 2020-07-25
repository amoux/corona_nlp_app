from typing import List, Optional

from pydantic import BaseModel, Field


class QuestionAnsweringBase(BaseModel):
    question: str = Field(None, title="Question to pass to the model.",
                          max_length=2000)


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
    context: str = Field(..., description=(
        "The context for answering the input question."))


class QuestionAnsweringWithContextOutput(BaseModel):
    answer: str
    context: str


class SentenceSimilarityInput(BaseModel):
    sentence: str
    topk: Optional[int] = 5
    nprobe: Optional[int] = 1
    add_paper_ids: Optional[bool] = Field(None, description=(
        "Weather to include the paper ids with the output."))


class SentenceSimilarityOutput(BaseModel):
    n_sents: int
    sents: List[str]
    dists: List[int]


class SentenceSimilarityWithPaperIdsOutput(SentenceSimilarityOutput):
    paper_ids: List[int]


class TextToSpeechIn(BaseModel):
    pass
