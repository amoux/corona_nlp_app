import uvicorn
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from app.api.router import router as api_router
from app.utils import app_config


def corona_nlp_app() -> FastAPI:
    app = FastAPI()

    config = app_config()
    APP_HTTP = 'http://localhost'
    BACKEND_PORT = f"{APP_HTTP}:{config['fastapi']['port']}"
    FRONTEND_PORT = f"{APP_HTTP}:{config['streamlit']['port']}"

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[APP_HTTP, BACKEND_PORT, FRONTEND_PORT],
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*'],
    )
    app.include_router(api_router)
    return app


app = corona_nlp_app()
