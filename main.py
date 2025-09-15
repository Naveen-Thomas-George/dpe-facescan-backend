import io
import os
import zipfile
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.config import settings
from app.db import Base, engine, get_db
from app.models import Event, Photo
from app.services.face import embed_image_bytes
from app.services.index import load_or_create_index, search


app = FastAPI(title="CUBYCSPO API")

# Define allowed origins for your React frontend
origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://localhost:8080",  # <-- Add this line
    "http://192.168.56.1:8080",  # <-- Add this line

    "https://gentle-smoke-0f0a7e900.azurestaticapps.net"
]

# Add CORS middleware to allow communication with the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create database tables on startup
Base.metadata.create_all(bind=engine)


@app.post("/api/search")
async def api_search(
    selfie: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if selfie.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    img_bytes = await selfie.read()
    emb = embed_image_bytes(img_bytes)
    if emb.sum() == 0.0:
        return {"matches": [], "note": "No face detected in the selfie."}

    index, ids = load_or_create_index(event_slug=settings.EVENT_SLUG)
    if index.ntotal == 0 or ids.size == 0:
        return {"matches": [], "note": "The photo index is empty. Please ingest photos first."}

    sims, idxs = search(index, emb, top_k=settings.TOP_K)
    
    # This section is a bit slow. Let's optimize it to run a single database query.
    matched_ids = [int(ids[i]) for i in idxs.tolist() if i >= 0]
    
    if not matched_ids:
        return {"matches": []}

    # Fetch all matching photos in one go
    photo_results = db.execute(
        select(Photo).where(Photo.id.in_(matched_ids))
    ).scalars().all()
    
    # Create a dictionary for quick lookups
    photos_by_id = {p.id: p for p in photo_results}

    matches = []
    for sim, i in zip(sims.tolist(), idxs.tolist()):
        if i < 0:
            continue
        
        photo_id = int(ids[i])
        photo = photos_by_id.get(photo_id)

        # Apply threshold and format the response
        if photo and (settings.FAISS_METRIC != "cosine" or sim >= settings.MATCH_THRESHOLD):
            matches.append({
                "photo_id": photo.id,
                "url": photo.uri,
                "thumb": photo.thumb_uri,
                "score": float(sim),
            })

    return {"matches": matches}


@app.post("/download_zip")
async def download_zip(request: Request):
    data = await request.json()
    photo_urls = data.get("urls", [])

    if not photo_urls:
        return JSONResponse({"error": "No photo URLs provided."}, status_code=400)

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        for i, url in enumerate(photo_urls):
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    # Get the file extension from the URL
                    file_extension = os.path.splitext(url)[1] or ".jpg"
                    zf.writestr(f"photo_{i+1}{file_extension}", response.content)
            except Exception as e:
                print(f"Error fetching {url}: {e}")

    zip_buffer.seek(0)

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=matched_photos.zip"},
    )


@app.get("/healthz")
async def healthz():
    """A simple health check endpoint."""
    return {"status": "ok"}