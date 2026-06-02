from uuid import UUID

# Shared in-memory store for real-time video processing progress
# Stored as { video_id (UUID): progress_pct (int 0-100) }
# Lives in a separate module to avoid circular imports between main.py and video_processing.py

processing_progress: dict[UUID, int] = {}
cancelled_tasks: set[UUID] = set()

