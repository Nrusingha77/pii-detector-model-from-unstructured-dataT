from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from detect_pii import router as pii_router
from file_handler import router as file_router
from db.mongodb_conn import get_database

app = FastAPI(
    title="PII Detection API",
    description="API for detecting Personal Identifiable Information in text",
    version="1.0.0"
)

# Dependency to get database instance
async def get_db():
    db = get_database()
    try:
        yield db
    finally:
        db.client.close()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(
    pii_router,
    prefix="/pii",
    tags=["PII Detection"]
)
app.include_router(
    file_router,
    prefix="/files",
    tags=["File Processing"]
)

@app.get("/")
async def root():
    return {
        "status": "FastAPI server is running",
        "version": "1.0.0",
        "docs_url": "/docs"
    }
