from fastapi import APIRouter

from app.api.v1 import question_answering, sentence_similarity, text_to_speech

router = APIRouter()
router.include_router(question_answering.router, tags=['answer'])
router.include_router(sentence_similarity.router, tags=['similar'])
#router.include_router(text_to_speech.router, tags=['audio_file'])
