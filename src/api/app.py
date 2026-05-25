from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from src.api.routes import router
from src.api.dependencies import get_rag_pipeline
from contextlib import asynccontextmanager

# Path to the UI directory
UI_DIR = Path(__file__).resolve().parent.parent / "ui"

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Startup] Start models (Warm-up)...")
    get_rag_pipeline()
    print("[Startup] Ready!")
    yield
    print("[Shutdown] Cleaning up resources...")

def create_app() -> FastAPI:
    app = FastAPI(
        title="Medical RAG API",
        description="REST API for Medical RAG",
        version="1.0.0",
        lifespan=lifespan
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API routes
    app.include_router(router)

    # Serve static assets (CSS, JS)
    app.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")

    # Serve the main UI page at root
    @app.get("/", include_in_schema=False)
    async def serve_ui():
        return FileResponse(str(UI_DIR / "index.html"))

    return app

app = create_app()