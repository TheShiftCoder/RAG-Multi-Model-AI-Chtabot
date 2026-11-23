"""Main FastAPI Application Entrypoint.

Configures middleware, routers, exception handlers, and database initialization.
"""

import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from multimodal_ai_companion.backend.core.config import settings
from multimodal_ai_companion.backend.database.session import engine, Base
import multimodal_ai_companion.backend.models  # Ensures all models are registered

# Import API Routers
from multimodal_ai_companion.backend.api.chat import router as chat_router
from multimodal_ai_companion.backend.api.documents import router as documents_router
from multimodal_ai_companion.backend.api.audio import router as audio_router
from multimodal_ai_companion.backend.api.images import router as images_router
from multimodal_ai_companion.backend.api.feedback import router as feedback_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("multimodal_ai_companion")

# Initialize database schema tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    description="Functional MVP of Secure Multimodal AI Companion with RAG Architecture.",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware with restricted origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Safe exception handler preventing internal stack trace disclosure."""
    logger.error(f"Unhandled server error on {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred. Please try again later."}
    )


# Health check endpoint
@app.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint returning system status."""
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "database": engine.url.drivername,
        "environment": settings.ENVIRONMENT,
    }


# Include Routers under API_V1_STR
app.include_router(chat_router, prefix=settings.API_V1_STR)
app.include_router(documents_router, prefix=settings.API_V1_STR)
app.include_router(audio_router, prefix=settings.API_V1_STR)
app.include_router(images_router, prefix=settings.API_V1_STR)
app.include_router(feedback_router, prefix=settings.API_V1_STR)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

