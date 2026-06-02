import uuid
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import datetime
from pgvector.sqlalchemy import Vector
from database import Base


class Operator(Base):
    __tablename__ = "operators"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String, default="")
    role = Column(String, default="operator")   # operator | admin
    ai_model = Column(String, default="yolov8n.pt") # Assigned AI Model (e.g. yolov8n.pt, rtdetr-l.pt)
    min_confidence = Column(Float, default=0.45)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class Video(Base):
    __tablename__ = "videos"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    filename = Column(String, index=True)
    filepath = Column(String)
    duration = Column(Float, default=0.0)
    status = Column(String, default="uploading") # uploading, processing, completed, failed
    process_tags = Column(Boolean, default=True)
    process_ocr = Column(Boolean, default=True)
    process_semantic = Column(Boolean, default=True)
    processing_time_sec = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    operator_id = Column(UUID(as_uuid=True), ForeignKey("operators.id"), nullable=True)

    detections = relationship("Detection", back_populates="video", cascade="all, delete-orphan")

class Detection(Base):
    __tablename__ = "detections"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    video_id = Column(UUID(as_uuid=True), ForeignKey("videos.id"))
    track_id = Column(Integer, nullable=True, index=True)
    timestamp_sec = Column(Float)
    object_type = Column(String, index=True)
    confidence = Column(Float)
    bbox_json = Column(JSON)
    bbox_center_x = Column(Float, nullable=True)
    bbox_center_y = Column(Float, nullable=True)
    bbox_width = Column(Float, nullable=True)
    bbox_height = Column(Float, nullable=True)
    segmentation_json = Column(JSON, nullable=True)  # Store normalized polygon points: [[x1, y1], [x2, y2], ...]
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    video = relationship("Video", back_populates="detections")

class TargetClass(Base):
    __tablename__ = "target_classes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    is_enabled = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class LicencePlate(Base):
    __tablename__ = "licence_plates"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    video_id = Column(UUID(as_uuid=True), ForeignKey("videos.id"), nullable=False)
    detection_id = Column(UUID(as_uuid=True), ForeignKey("detections.id"), nullable=True)
    timestamp_sec = Column(Float, nullable=False)
    plate_number = Column(String(20), nullable=False, index=True)
    confidence = Column(Float)
    bbox_json = Column(JSON)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    video = relationship("Video")

class FrameEmbedding(Base):
    __tablename__ = "frame_embeddings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    video_id = Column(UUID(as_uuid=True), ForeignKey("videos.id"), nullable=False)
    timestamp_sec = Column(Float, nullable=False)
    embedding = Column(Vector(768), nullable=False)  # SigLIP features are 768
    caption = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    video = relationship("Video")
