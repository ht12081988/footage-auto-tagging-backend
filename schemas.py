from pydantic import BaseModel, EmailStr
from typing import List, Optional
from datetime import datetime
from typing import Any
from uuid import UUID


# ── Operator / Auth schemas ────────────────────────────────────────────────────

class OperatorCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: Optional[str] = ""
    role: Optional[str] = "operator"
    min_confidence: Optional[float] = 0.45

class OperatorLogin(BaseModel):
    email: str
    password: str

class TargetClassUpdate(BaseModel):
    is_enabled: bool

class TargetClassResponse(BaseModel):
    id: UUID
    name: str
    is_enabled: bool
    created_at: datetime

    class Config:
        from_attributes = True

class OperatorResponse(BaseModel):
    id: UUID
    email: str
    full_name: str
    role: str
    ai_model: Optional[str] = "yolov8n.pt"
    min_confidence: Optional[float] = 0.45
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True

class LoginResponse(BaseModel):
    message: str
    operator: OperatorResponse

class DetectionBase(BaseModel):
    timestamp_sec: float
    object_type: str
    confidence: float
    bbox_json: Any
    track_id: Optional[int] = None
    bbox_center_x: Optional[float] = None
    bbox_center_y: Optional[float] = None
    bbox_width: Optional[float] = None
    bbox_height: Optional[float] = None
    segmentation_json: Optional[Any] = None

class Detection(DetectionBase):
    id: UUID
    video_id: UUID
    created_at: datetime

    class Config:
        from_attributes = True

class VideoBase(BaseModel):
    filename: str
    operator_id: Optional[UUID] = None

class Video(VideoBase):
    id: UUID
    filename: str
    filepath: str
    duration: float
    status: str
    process_tags: bool
    process_ocr: bool
    process_semantic: bool
    processing_time_sec: float
    created_at: datetime
    operator_id: Optional[UUID] = None
    detections: List[Detection] = []

    class Config:
        from_attributes = True

class LicencePlateResponse(BaseModel):
    id: UUID
    video_id: UUID
    timestamp_sec: float
    plate_number: str
    confidence: Optional[float] = None
    created_at: datetime
    video_filename: Optional[str] = None
    video_filepath: str

    class Config:
        from_attributes = True

class FrameCaptionResponse(BaseModel):
    id: UUID
    video_id: UUID
    timestamp_sec: float
    caption: str
    created_at: datetime

    class Config:
        from_attributes = True
