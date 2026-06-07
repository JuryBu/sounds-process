import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import api_router


app = FastAPI(
    title="BirdVoice API",
    version="0.1.0",
    description="BirdVoice 鸟类声音识别系统后端",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5177",
        "http://127.0.0.1:5177",
        *[origin.strip() for origin in os.getenv("BIRDVOICE_CORS_ORIGINS", "").split(",") if origin.strip()],
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")
