import io
import numpy as np
import logging
import cv2
from insightface.app import FaceAnalysis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("face-service")


class FaceEngine:
    """Singleton for InsightFace engine."""
    app = None
    loaded = False

    @classmethod
    def load(cls):
        if cls.loaded:
            return
        try:
            logger.info("Loading InsightFace (buffalo_l)...")
            cls.app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            cls.app.prepare(ctx_id=0, det_size=(640, 640))
            cls.loaded = True
            logger.info("InsightFace model loaded successfully.")
        except Exception as e:
            logger.error(f"InsightFace failed to load: {e}")
            cls.app = None


def _bytes_to_bgr(image_bytes: bytes):
    """Decode bytes -> OpenCV BGR ndarray"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        logger.error("cv2.imdecode failed to decode image bytes.")
    return img


def extract_face_embeddings(image_bytes: bytes):
    """Return list of (embedding, bbox)."""
    FaceEngine.load()
    if FaceEngine.app is None:
        logger.error("InsightFace engine not available.")
        return []

    img = _bytes_to_bgr(image_bytes)
    if img is None:
        return []

    faces = FaceEngine.app.get(img)
    logger.info(f"InsightFace detected {len(faces)} faces.")

    results = []
    for f in faces:
        emb = np.asarray(f.normed_embedding, dtype="float32")
        x1, y1, x2, y2 = [int(v) for v in f.bbox]
        bbox = [x1, y1, max(0, x2 - x1), max(0, y2 - y1)]
        results.append((emb, bbox))

    return results


def embed_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Return the largest face embedding or zeros if none found."""
    faces = extract_face_embeddings(image_bytes)
    if not faces:
        logger.warning("No faces detected, returning zero embedding.")
        return np.zeros((512,), dtype="float32")

    # Choose the largest detected face
    def area(b):
        return max(1, b[2]) * max(1, b[3])
    faces.sort(key=lambda fb: area(fb[1]), reverse=True)

    emb = faces[0][0]
    logger.info(f"Returning embedding of shape {emb.shape}, dtype={emb.dtype}")
    return emb
