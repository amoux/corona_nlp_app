from fastapi import APIRouter

router = APIRouter()


@router.get('/')
def get_audio_file(text: str):
    pass
