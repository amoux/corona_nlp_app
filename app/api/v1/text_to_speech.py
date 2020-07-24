from fastapi import APIRouter

router = APIRouter()


@router.get('/')
async def get_audio_file(text: str):
    pass
