from fastapi import APIRouter

from app.api.v1 import question_answering, text_to_speech

router = APIRouter()
router.include_router(question_answering.router)
router.include_router(text_to_speech.router)
