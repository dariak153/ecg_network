from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.segmentation_api import router as seg_router

app = FastAPI(title="ECG QRS Segmentation Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(seg_router)
