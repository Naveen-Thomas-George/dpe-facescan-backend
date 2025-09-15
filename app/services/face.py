# app/services/face.py
import io
import numpy as np

# Primary: InsightFace
_INSIGHT_OK = True
try:
    import cv2
    from insightface.app import FaceAnalysis
except Exception:
    _INSIGHT_OK = False

# Fallback: DeepFace
_DEEPFACE_OK = True
try:
    from deepface import DeepFace
except Exception:
    _DEEPFACE_OK = False


class _FaceEngines:
    """Lazy loaders so import failures don't crash the app."""
    insight_app = None
    loaded = False

    @classmethod
    def load(cls):
        if cls.loaded:
            return
        if _INSIGHT_OK:
            try:
                # CPU-friendly config
                cls.insight_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
                cls.insight_app.prepare(ctx_id=0, det_size=(640, 640))
            except Exception:
                cls.insight_app = None
        cls.loaded = True


def _bytes_to_bgr(image_bytes: bytes):
    """Decode bytes -> OpenCV BGR ndarray"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def extract_face_embeddings(image_bytes: bytes):
    """
    Returns: list of tuples -> (embedding: np.ndarray[512], bbox: [x, y, w, h])
    - Uses InsightFace for multi-face detection/embeddings.
    - Falls back to DeepFace (ArcFace) if InsightFace unavailable.
    """
    results = []

    # ---------- Primary: InsightFace ----------
    if _INSIGHT_OK:
        try:
            _FaceEngines.load()
            if _FaceEngines.insight_app is not None:
                img = _bytes_to_bgr(image_bytes)
                if img is None:
                    raise ValueError("Unable to decode image bytes")
                faces = _FaceEngines.insight_app.get(img)  # list of Face objects
                for f in faces:
                    # normed_embedding is already L2-normalized float32 (512-D)
                    emb = np.asarray(f.normed_embedding, dtype="float32")
                    x1, y1, x2, y2 = [int(v) for v in f.bbox]
                    bbox = [x1, y1, max(0, x2 - x1), max(0, y2 - y1)]
                    results.append((emb, bbox))
                if results:
                    return results
        except Exception:
            pass  # fall through to fallback

    # ---------- Fallback: DeepFace ----------
    if _DEEPFACE_OK:
        try:
            # DeepFace.represent can return multiple faces with regions
            # Use ArcFace (512-D) to stay close to InsightFace dimension
            reps = DeepFace.represent(
                img_path=io.BytesIO(image_bytes),
                model_name="ArcFace",
                detector_backend="retinaface",
                enforce_detection=False
            )
            # DeepFace returns dict or list-of-dicts depending on #faces
            if isinstance(reps, dict):
                reps = [reps]
            for r in reps:
                if isinstance(r, dict):
                    emb = np.asarray(r.get("embedding", []), dtype="float32")
                    region = r.get("region", {}) or {}
                    x, y, w, h = int(region.get("x", 0)), int(region.get("y", 0)), int(region.get("w", 0)), int(region.get("h", 0))
                    if emb.size > 0:
                        results.append((emb, [x, y, w, h]))
            if results:
                # If ArcFace in DeepFace returns 512-D, great; otherwise we’ll cast to float32
                return [ (e.astype("float32"), b) for e,b in results ]
        except Exception:
            pass

    # ---------- No faces or libraries failed ----------
    return []


def embed_image_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Backwards-compat single-vector function:
    - If multiple faces found, return the **best (largest box)** face embedding.
    - If none found, return a zero vector (512-D) to avoid crashes.
    """
    faces = extract_face_embeddings(image_bytes)
    if not faces:
        return np.zeros((512,), dtype="float32")
    # choose face with largest bbox area
    def area(b): 
        return max(1, b[2]) * max(1, b[3])
    faces.sort(key=lambda fb: area(fb[1]), reverse=True)
    return faces[0][0]
