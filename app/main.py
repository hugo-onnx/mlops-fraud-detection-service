from fastapi import FastAPI
from gradio.routes import mount_gradio_app
from contextlib import asynccontextmanager

from app.api.routes import router
from app.services.ml_service import model_service
from app.ui.gradio_app import gr_app
from app.config.config import logger

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model_service.load_model()
    except Exception as e:
        logger.error(f"Model load failed: {e}")
    yield


app = FastAPI(title="Fraud Detection API", lifespan=lifespan)

app.include_router(router)

app = mount_gradio_app(app, gr_app, path="/ui")