from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Depends, HTTPException, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import shutil
import os
import uuid
from uuid import UUID
import httpx

# Simple manual helper to load environment variables from .env if present
def load_env_file():
    if os.path.exists(".env"):
        with open(".env", "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split("=", 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        val = parts[1].strip().strip('"').strip("'")
                        # Only load specific Vercel Blob keys to prevent touching DB configuration
                        if key in ["BLOB_READ_WRITE_TOKEN", "VERCEL_BLOB_TOKEN"]:
                            os.environ[key] = val

load_env_file()

# Clean up incorrect DATABASE_URL if it was injected from .env in a previous hot-reload
if os.environ.get("DATABASE_URL") == "postgresql://postgres:postgres@localhost:5432/footage_auto_tagging":
    del os.environ["DATABASE_URL"]

import models
import schemas
from database import engine, get_db
from video_processing import process_video_task, UPLOAD_DIR
import auth

from fastapi.staticfiles import StaticFiles

# Create database tables if they don't exist
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="CCTV Auto-Tagging API - Vigilant.ai")

# ── In-memory progress store (video_id -> 0-100) ─────────────────────────────
from progress_store import processing_progress, cancelled_tasks



@app.on_event("startup")
def seed_default_admin():
    """Create a default admin operator if no operators exist yet."""
    from database import SessionLocal
    db = SessionLocal()
    try:
        existing = db.query(models.Operator).first()
        if not existing:
            auth.create_operator(
                db,
                email="admin@sentinel.sys",
                password="sentinel123",
                full_name="Default Admin",
                role="admin",
            )
            print("[SEED] Default admin created: admin@sentinel.sys / sentinel123")
            
        # Seed 80 COCO classes
        coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
            'hair drier', 'toothbrush'
        ]
        
        existing_targets = db.query(models.TargetClass).count()
        if existing_targets == 0:
            default_active = ['person', 'car', 'backpack', 'handbag', 'bicycle', 'cow', 'traffic light']
            new_targets = []
            for cls in coco_classes:
                new_targets.append(models.TargetClass(
                    name=cls,
                    is_enabled=(cls in default_active)
                ))
            db.bulk_save_objects(new_targets)
            db.commit()
            print("[SEED] 80 default COCO target classes seeded.")
            
    finally:
        db.close()

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Auth Routes ─────────────────────────────────────────────────────────────

@app.post("/auth/register", response_model=schemas.OperatorResponse)
def register_operator(payload: schemas.OperatorCreate, db: Session = Depends(get_db)):
    """Register a new operator account."""
    existing = auth.get_operator_by_email(db, payload.email)
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered.")
    op = auth.create_operator(
        db,
        email=payload.email,
        password=payload.password,
        full_name=payload.full_name or "",
        role=payload.role or "operator",
    )
    return op


@app.post("/auth/login", response_model=schemas.LoginResponse)
def login_operator(payload: schemas.OperatorLogin, db: Session = Depends(get_db)):
    """Authenticate an operator and return their profile."""
    op = auth.authenticate_operator(db, payload.email, payload.password)
    if not op:
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    return {"message": "Login successful", "operator": op}


@app.get("/auth/operators", response_model=list[schemas.OperatorResponse])
def list_operators(db: Session = Depends(get_db)):
    """List all registered operators (admin use)."""
    return db.query(models.Operator).all()


# ── Video Routes ─────────────────────────────────────────────────────────────

async def upload_to_vercel_blob_stream(file: UploadFile, filename: str) -> str:
    # Ensure environment variables are loaded
    token = (os.environ.get("BLOB_READ_WRITE_TOKEN") or "").strip()
    if not token:
        load_env_file()
        token = (os.environ.get("BLOB_READ_WRITE_TOKEN") or "").strip()
        
    if not token:
        raise HTTPException(
            status_code=400,
            detail="BLOB_READ_WRITE_TOKEN is not configured in backend .env file. Please add it to start uploading to Vercel Blob."
        )
    
    url = f"https://blob.vercel-storage.com/{filename}"
    headers = {
        "Authorization": f"Bearer {token}",
        "x-api-version": "7",
        "Content-Type": "video/mp4"
    }
    
    async def file_generator():
        await file.seek(0)
        while True:
            chunk = await file.read(1024 * 1024)  # 1MB chunk size
            if not chunk:
                break
            yield chunk

    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            response = await client.put(url, content=file_generator(), headers=headers)
            if response.status_code != 200:
                raise HTTPException(
                    status_code=500,
                    detail=f"Vercel Blob API error ({response.status_code}): {response.text}"
                )
            data = response.json()
            return data["url"]
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Network error while transmitting to Vercel Blob: {exc}"
        )


@app.post("/upload", response_model=schemas.Video)
async def upload_video(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...), 
    operator_id: UUID = Form(...),
    process_tags: bool = Form(True),
    process_ocr: bool = Form(True),
    process_semantic: bool = Form(True),
    db: Session = Depends(get_db)
):
    if not file.filename.endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Only MP4 files are supported.")
    
    unique_filename = f"{uuid.uuid4()}_{file.filename}"
    
    # Stream the file directly to Vercel Blob
    blob_url = await upload_to_vercel_blob_stream(file, unique_filename)
        
    new_video = models.Video(
        filename=file.filename,
        filepath=blob_url,
        status="uploading",
        operator_id=operator_id,
        process_tags=process_tags,
        process_ocr=process_ocr,
        process_semantic=process_semantic
    )
    db.add(new_video)
    db.commit()
    db.refresh(new_video)
    
    background_tasks.add_task(process_video_task, new_video.id, blob_url)
    
    return new_video

@app.post("/videos/{video_id}/reprocess")
def reprocess_video(
    video_id: UUID, 
    background_tasks: BackgroundTasks,
    process_tags: bool = Form(...),
    process_ocr: bool = Form(...),
    process_semantic: bool = Form(...),
    operator_id: UUID = Form(...),
    db: Session = Depends(get_db)
):
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if video.operator_id != operator_id:
        raise HTTPException(status_code=403, detail="Access denied")
        
    video.process_tags = process_tags
    video.process_ocr = process_ocr
    video.process_semantic = process_semantic
    video.status = "processing"
    
    # Delete old data for the modules we are about to re-run
    if process_tags:
        db.query(models.Detection).filter(models.Detection.video_id == video_id).delete()
    elif process_ocr:
        db.query(models.LicencePlate).filter(models.LicencePlate.video_id == video_id).delete()
        
    if process_semantic:
        db.query(models.FrameEmbedding).filter(models.FrameEmbedding.video_id == video_id).delete()
        
    db.commit()
    
    background_tasks.add_task(process_video_task, video.id, video.filepath)
    return {"message": "Reprocessing started"}

@app.get("/videos", response_model=list[schemas.Video])
def get_videos(operator_id: UUID = Query(...), skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    videos = db.query(models.Video).filter(models.Video.operator_id == operator_id).order_by(models.Video.created_at.desc()).offset(skip).limit(limit).all()
    return videos

@app.get("/videos/{video_id}", response_model=schemas.Video)
def get_video(video_id: UUID, operator_id: UUID = Query(...), db: Session = Depends(get_db)):
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if video.operator_id != operator_id:
        raise HTTPException(status_code=403, detail="Access denied")
    return video

@app.get("/videos/{video_id}/detections", response_model=list[schemas.Detection])
def get_video_detections(video_id: UUID, operator_id: UUID = Query(...), db: Session = Depends(get_db)):
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if video.operator_id != operator_id:
        raise HTTPException(status_code=403, detail="Access denied")
    detections = db.query(models.Detection).filter(models.Detection.video_id == video_id).order_by(models.Detection.timestamp_sec).all()
    return detections

@app.get("/videos/{video_id}/plates", response_model=list[schemas.LicencePlateResponse])
def get_video_plates(video_id: UUID, operator_id: UUID = Query(...), db: Session = Depends(get_db)):
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if video.operator_id != operator_id:
        raise HTTPException(status_code=403, detail="Access denied")
    plates = db.query(models.LicencePlate).filter(models.LicencePlate.video_id == video_id).order_by(models.LicencePlate.timestamp_sec).all()
    results = []
    for p in plates:
        results.append(schemas.LicencePlateResponse(
            id=p.id,
            video_id=p.video_id,
            timestamp_sec=p.timestamp_sec,
            plate_number=p.plate_number,
            confidence=p.confidence,
            created_at=p.created_at,
            video_filename=p.video.filename if p.video else "Unknown",
            video_filepath=p.video.filepath if p.video else ""
        ))
    return results

@app.get("/videos/{video_id}/captions", response_model=list[schemas.FrameCaptionResponse])
def get_video_captions(video_id: UUID, operator_id: UUID = Query(...), db: Session = Depends(get_db)):
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if video.operator_id != operator_id:
        raise HTTPException(status_code=403, detail="Access denied")
    embeddings = db.query(models.FrameEmbedding).filter(models.FrameEmbedding.video_id == video_id).order_by(models.FrameEmbedding.timestamp_sec).all()
    return [
        schemas.FrameCaptionResponse(
            id=r.id,
            video_id=r.video_id,
            timestamp_sec=r.timestamp_sec,
            caption=r.caption or "",
            created_at=r.created_at
        )
        for r in embeddings
    ]

@app.get("/videos/{video_id}/progress")
def get_video_progress(video_id: UUID):
    """Returns the live processing progress for a video (0-100)."""
    return {"progress_pct": processing_progress.get(video_id, 0)}

@app.post("/videos/{video_id}/cancel")
def cancel_video_processing(video_id: UUID, db: Session = Depends(get_db)):
    """Add a video_id to cancelled_tasks to abort processing gracefully."""
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    
    if video.status not in ["processing", "uploading"]:
        raise HTTPException(status_code=400, detail="Video is not currently processing or uploading")

    # Add to cancelled set
    cancelled_tasks.add(video_id)
    
    # Mark status as cancelled in database immediately so the frontend updates instantly
    video.status = "cancelled"
    db.commit()
    
    print(f"[API] Cancellation request registered for video ID {video_id}.")
    return {"message": "Processing cancellation initiated"}


@app.get("/search", response_model=list[dict])
def search_detections(object: str, operator_id: UUID = Query(...), db: Session = Depends(get_db)):
    detections = db.query(models.Detection).join(models.Video).filter(models.Detection.object_type.ilike(f"%{object}%")).filter(models.Video.operator_id == operator_id).order_by(models.Detection.timestamp_sec).all()
    
    results = []
    for d in detections:
        results.append({
            "video_id": d.video_id,
            "timestamp": d.timestamp_sec,
            "confidence": d.confidence,
            "object_type": d.object_type,
            "video_filename": d.video.filename if d.video else "Unknown"
        })
    return results

@app.get("/search/semantic")
def search_semantic(
    query: str = Query(..., min_length=1), 
    operator_id: UUID = Query(...),
    video_id: UUID = Query(None, description="Optional single video ID to search inside"),
    limit: int = Query(10, le=100), 
    threshold: float = Query(0.98, description="Maximum cosine distance (lower means stricter matching)"),
    db: Session = Depends(get_db)
):
    """Search frames by semantic meaning using SigLIP/BLIP vector embeddings."""
    try:
        from vector_search import vector_service
    except ImportError:
        raise HTTPException(status_code=501, detail="Vector search service not available")

    # Generate vector for text query
    try:
        query_vector = vector_service.get_text_embedding(query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate text embedding: {e}")
    
    # Debug: log exact distances of top 5 closest frame embeddings to understand distribution
    try:
        from sqlalchemy import text
        debug_query = db.query(models.FrameEmbedding.id, models.FrameEmbedding.embedding.cosine_distance(query_vector).label("dist"))\
            .join(models.Video)
        if video_id is not None:
            debug_query = debug_query.filter(models.FrameEmbedding.video_id == video_id)
        else:
            debug_query = debug_query.filter(models.Video.operator_id == operator_id)
        debug_results = debug_query.order_by("dist").limit(5).all()
        print(f"[Semantic Search Debug] Query: '{query}' | Top 5 distances: {[round(r.dist, 4) for r in debug_results]}")
    except Exception as de:
        print(f"[Semantic Search Debug] Failed to log debug distances: {de}")

    # Query database selecting the model and calculating the cosine distance
    distance_expr = models.FrameEmbedding.embedding.cosine_distance(query_vector)
    query_base = db.query(models.FrameEmbedding, distance_expr.label("distance")).join(models.Video)
    
    if video_id is not None:
        query_base = query_base.filter(models.FrameEmbedding.video_id == video_id)
    else:
        query_base = query_base.filter(models.Video.operator_id == operator_id)
        
    results = query_base.filter(distance_expr < threshold)\
        .order_by(distance_expr)\
        .limit(limit)\
        .all()
        
    # Fallback to looser threshold of 0.98 if requested threshold was too strict and returned no results
    if not results and threshold < 0.98:
        print(f"[Semantic Search] No results under strict threshold {threshold}. Retrying with loose threshold 0.98...")
        query_base_fallback = db.query(models.FrameEmbedding, distance_expr.label("distance")).join(models.Video)
        if video_id is not None:
            query_base_fallback = query_base_fallback.filter(models.FrameEmbedding.video_id == video_id)
        else:
            query_base_fallback = query_base_fallback.filter(models.Video.operator_id == operator_id)
            
        results = query_base_fallback.filter(distance_expr < 0.98)\
            .order_by(distance_expr)\
            .limit(limit)\
            .all()
        
    # Scaled confidence calculation
    response_data = []
    for r, distance in results:
        similarity = 1.0 - float(distance)
        min_sim = 0.05
        max_sim = 0.15
        scaled_conf = 0.0
        if similarity > min_sim:
            scaled_conf = min(1.0, (similarity - min_sim) / (max_sim - min_sim))
            
        response_data.append({
            "id": r.id,
            "video_id": r.video_id,
            "timestamp": r.timestamp_sec,
            "caption": r.caption,
            "video_filename": r.video.filename if r.video else "Unknown",
            "video_filepath": r.video.filepath if r.video else "",
            "confidence": scaled_conf
        })
        
    return response_data

@app.get("/plates/search", response_model=list[schemas.LicencePlateResponse])
def search_plates(
    q: str = Query(..., min_length=1), 
    operator_id: UUID = Query(...), 
    video_id: UUID = Query(None, description="Optional video ID to search inside"),
    limit: int = Query(10, le=200), 
    db: Session = Depends(get_db)
):
    """Search licence plates by partial match. Returns up to `limit` results."""
    query_base = db.query(models.LicencePlate).join(models.Video)
    if video_id is not None:
        query_base = query_base.filter(models.LicencePlate.video_id == video_id)
    else:
        query_base = query_base.filter(models.Video.operator_id == operator_id)
        
    plates = (
        query_base.filter(models.LicencePlate.plate_number.ilike(f"%{q}%"))
        .order_by(models.LicencePlate.created_at.desc())
        .limit(limit)
        .all()
    )
    results = []
    for p in plates:
        results.append(schemas.LicencePlateResponse(
            id=p.id,
            video_id=p.video_id,
            timestamp_sec=p.timestamp_sec,
            plate_number=p.plate_number,
            confidence=p.confidence,
            created_at=p.created_at,
            video_filename=p.video.filename if p.video else "Unknown",
            video_filepath=p.video.filepath if p.video else ""
        ))
    return results

@app.get("/targets", response_model=list[schemas.TargetClassResponse])
def get_targets(db: Session = Depends(get_db)):
    return db.query(models.TargetClass).order_by(models.TargetClass.name).all()

@app.patch("/targets/{target_name}", response_model=schemas.TargetClassResponse)
def update_target(target_name: str, payload: schemas.TargetClassUpdate, db: Session = Depends(get_db)):
    target = db.query(models.TargetClass).filter(models.TargetClass.name == target_name).first()
    if not target:
        raise HTTPException(status_code=404, detail="Target class not found")
    
    target.is_enabled = payload.is_enabled
    db.commit()
    db.refresh(target)
    return target

@app.delete("/videos/{video_id}")
def delete_video(video_id: UUID, operator_id: UUID = Query(...), db: Session = Depends(get_db)):
    video = db.query(models.Video).filter(models.Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if video.operator_id != operator_id:
        raise HTTPException(status_code=403, detail="Access denied")

    # Delete the physical file from disk
    if os.path.exists(video.filepath):
        try:
            os.remove(video.filepath)
        except Exception as e:
            print(f"Failed to delete file {video.filepath}: {e}")

    # Delete all child records that don't have ORM cascade set up
    db.query(models.FrameEmbedding).filter(models.FrameEmbedding.video_id == video_id).delete()
    db.query(models.LicencePlate).filter(models.LicencePlate.video_id == video_id).delete()
    # Detections cascade via ORM (cascade="all, delete-orphan") but delete explicitly to be safe
    db.query(models.Detection).filter(models.Detection.video_id == video_id).delete()

    db.delete(video)
    db.commit()
    return {"message": "Video permanently deleted"}
