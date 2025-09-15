# scripts/ingest_photos.py
import argparse
import os
import uuid
import numpy as np
import hashlib
from sqlalchemy.orm import Session
from sqlalchemy import select
from glob import glob

from app.config import settings
from app.db import SessionLocal
from app.models import Event, Photo
from app.services.storage import upload_image_bytes, upload_file
from app.services.face import extract_face_embeddings, embed_image_bytes
from app.services.index import load_or_create_index, add_embeddings, persist_index, _idx, _ids

def compute_file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def find_all_images(folder: str):
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.webp")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob(os.path.join(folder, "**", ext), recursive=True))
    return image_paths

def main(event_slug: str, folder: str):
    # Get or create the event with a short-lived database session
    with SessionLocal() as db:
        ev = db.execute(select(Event).where(Event.slug == event_slug)).scalar_one_or_none()
        if not ev:
            ev = Event(slug=event_slug, name=event_slug)
            db.add(ev)
            db.commit()
            db.refresh(ev)
        event_id = ev.id

    index, ids = load_or_create_index(event_slug=event_slug)
    new_embs, new_ids = [], []

    files = find_all_images(folder)
    print(f"Found {len(files)} images across all subfolders.")

    for path in files:
        with open(path, "rb") as f:
            data = f.read()

        file_hash = compute_file_hash(data)
        
        # --- NEW LOGIC: Do ALL heavy processing BEFORE touching the database ---
        
        # 1. Heavy AI processing for embeddings
        rep_emb = embed_image_bytes(data)
        faces = extract_face_embeddings(data)

        # 2. Network operation for upload
        subfolder_name = os.path.basename(os.path.dirname(path))
        file_extension = os.path.splitext(path)[1]
        public_id = f"{subfolder_name}/{uuid.uuid4().hex}{file_extension}"
        uri, thumb = upload_image_bytes(data, public_id)
        
        # --- NOW, open a very short database session just to save the results ---
        with SessionLocal() as db:
            if db.execute(select(Photo).where(Photo.file_hash == file_hash)).scalar_one_or_none():
                print(f"[SKIP] Duplicate already ingested → {path}")
                continue
            
            rep_path = os.path.join(settings.MEDIA_ROOT, "embeddings", f"{uuid.uuid4().hex}.npy")
            np.save(rep_path, rep_emb)
            
            photo = Photo(event_id=event_id, uri=uri, thumb_uri=thumb, embedding_path=rep_path, file_hash=file_hash)
            db.add(photo)
            db.commit()
            db.refresh(photo)
            photo_id = photo.id

        # Add the embeddings we found earlier to our list for indexing
        if not faces and rep_emb.size > 0:
            new_embs.append(rep_emb.astype("float32"))
            new_ids.append(photo_id)
            print(f"[NoFaces] {os.path.basename(path)} → using representative embedding.")
            continue
            
        count = 0
        for emb, _ in faces:
            if emb is not None and emb.size > 0:
                new_embs.append(emb.astype("float32"))
                new_ids.append(photo_id)
                count += 1
        
        print(f"[OK] {os.path.basename(path)} → {count} face(s) indexed → {uri}")

    # After the loop, persist and upload the index
    if new_embs:
        embs = np.vstack(new_embs).astype("float32")
        new_ids_arr = np.array(new_ids, dtype="int64")
        index, ids = add_embeddings(index, ids, embs, new_ids_arr, metric=settings.FAISS_METRIC)
        persist_index(index, ids, event_slug=event_slug)
        print(f"Indexed {len(new_ids)} new face embeddings locally.")

        print("Uploading index files to Azure Blob Storage...")
        index_file_path = _idx(event_slug)
        ids_file_path = _ids(event_slug)
        
        upload_file(local_path=index_file_path, public_id=os.path.basename(index_file_path), container_name=settings.AZURE_INDEX_CONTAINER)
        upload_file(local_path=ids_file_path, public_id=os.path.basename(ids_file_path), container_name=settings.AZURE_INDEX_CONTAINER)
    else:
        print("No new embeddings to index.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", required=True, help="event slug, e.g., christ-sports-2025")
    parser.add_argument("folder", help="The root folder containing all event images")
    args = parser.parse_args()
    main(args.event, args.folder)