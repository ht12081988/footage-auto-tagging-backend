import cv2
import numpy as np
import time
from ultralytics import YOLO
import json
import os
import re
from sqlalchemy.orm import Session
import shutil
import uuid
from uuid import UUID
import models
from database import SessionLocal
from progress_store import processing_progress, cancelled_tasks

from PIL import Image
try:
    from vector_search import vector_service
except Exception as ie:
    import traceback
    with open("debug_error.log", "a") as f:
        f.write(f"\n--- IMPORT ERROR IN VIDEO_PROCESSING ---\n{traceback.format_exc()}\n")
    vector_service = None

# ── RF-DETR Wrapper & Emulation classes ──────────────────────────────────────
class MockBox:
    def __init__(self, cls_id, conf, xyxy, xyxyn, track_id=None):
        self.cls = [cls_id]
        self.conf = [conf]
        self.xyxy = [xyxy]
        self.xyxyn = [xyxyn]
        self.id = [track_id] if track_id is not None else None

class MockMask:
    def __init__(self, xyn):
        self.xyn = xyn

class MockResult:
    def __init__(self, boxes, masks, names):
        self.boxes = boxes
        self.masks = masks
        self.names = names

class RFDETRWrapper:
    def __init__(self, model_name="rfdetr-large"):
        # Lazy imports of Segmentation models inside classes to keep application load fast
        from rfdetr import RFDETRSegLarge, RFDETRSegMedium, RFDETRSegSmall, RFDETRSegNano
        from rfdetr.assets.coco_classes import COCO_CLASSES
        
        self.model_name = model_name
        if "large" in model_name.lower():
            self.model = RFDETRSegLarge()
        elif "medium" in model_name.lower():
            self.model = RFDETRSegMedium()
        elif "small" in model_name.lower():
            self.model = RFDETRSegSmall()
        else:
            self.model = RFDETRSegNano()
            
        # Convert all default class names to lowercase for robust target class matching
        if isinstance(COCO_CLASSES, dict):
            self.names = {int(k): str(v).lower() for k, v in COCO_CLASSES.items()}
        else:
            self.names = {i: str(name).lower() for i, name in enumerate(COCO_CLASSES)}

    def track(self, frame, persist=True, tracker="bytetrack.yaml", conf=0.45, verbose=False):
        # Run inference using RF-DETR
        detections = self.model.predict(frame, threshold=conf)
        
        boxes_list = []
        masks_xyn = []
        h_f, w_f = frame.shape[:2]
        
        for idx in range(len(detections.xyxy)):
            box = detections.xyxy[idx]
            cls = int(detections.class_id[idx])
            score = float(detections.confidence[idx])
            mask = detections.mask[idx] if (detections.mask is not None and idx < len(detections.mask)) else None
            
            x1, y1, x2, y2 = box
            xyxy_float = [float(x1), float(y1), float(x2), float(y2)]
            xyxyn = [float(x1) / w_f, float(y1) / h_f, float(x2) / w_f, float(y2) / h_f]
            
            # Since RF-DETR is NMS-free and doesn't run a multi-frame tracker out of the box,
            # we supply None for track_id, which the pipeline and database handle perfectly.
            boxes_list.append(MockBox(cls, score, xyxy_float, xyxyn, track_id=None))
            
            if mask is not None:
                # RF-DETR can return masks either in full-frame space or as bbox-local crops.
                # Normalize all contour points back into the processed frame coordinate space.
                mask_array = np.asarray(mask)
                if mask_array.ndim == 3:
                    mask_array = np.squeeze(mask_array)
                mask_uint8 = (mask_array * 255).astype(np.uint8)
                if mask_uint8.ndim != 2:
                    masks_xyn.append([])
                    continue

                mask_h, mask_w = mask_uint8.shape[:2]

                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    epsilon = 0.002 * cv2.arcLength(largest_contour, True)
                    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                    contour_points = approx.reshape(-1, 2).astype(float)
                    box_w = max(float(x2 - x1), 1.0)
                    box_h = max(float(y2 - y1), 1.0)
                    contour_min = contour_points.min(axis=0)
                    contour_max = contour_points.max(axis=0)
                    contour_w = max(float(contour_max[0] - contour_min[0]), 1.0)
                    contour_h = max(float(contour_max[1] - contour_min[1]), 1.0)

                    def clamp_point(px, py):
                        return [
                            min(max(float(px), 0.0), float(w_f)),
                            min(max(float(py), 0.0), float(h_f)),
                        ]

                    def polygon_bounds(poly):
                        xs = [p[0] for p in poly]
                        ys = [p[1] for p in poly]
                        return [min(xs), min(ys), max(xs), max(ys)]

                    def box_iou(a, b):
                        ix1 = max(a[0], b[0])
                        iy1 = max(a[1], b[1])
                        ix2 = min(a[2], b[2])
                        iy2 = min(a[3], b[3])
                        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
                        area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
                        union = area_a + area_b - inter
                        return inter / union if union > 0 else 0.0

                    full_frame_poly = [
                        clamp_point(pt[0] * w_f / max(float(mask_w), 1.0), pt[1] * h_f / max(float(mask_h), 1.0))
                        for pt in contour_points
                    ]
                    bbox_local_poly = [
                        clamp_point(float(x1) + pt[0] * box_w / max(float(mask_w), 1.0), float(y1) + pt[1] * box_h / max(float(mask_h), 1.0))
                        for pt in contour_points
                    ]
                    contour_to_bbox_poly = [
                        clamp_point(
                            float(x1) + (pt[0] - contour_min[0]) * box_w / contour_w,
                            float(y1) + (pt[1] - contour_min[1]) * box_h / contour_h,
                        )
                        for pt in contour_points
                    ]

                    detection_bounds = [float(x1), float(y1), float(x2), float(y2)]
                    candidate_polys = [full_frame_poly, bbox_local_poly, contour_to_bbox_poly]
                    best_poly = max(
                        candidate_polys,
                        key=lambda poly: box_iou(polygon_bounds(poly), detection_bounds),
                    )
                    poly_points = [[p[0] / w_f, p[1] / h_f] for p in best_poly]
                    masks_xyn.append(poly_points)
                else:
                    masks_xyn.append([])
            else:
                masks_xyn.append([])
                
        return [MockResult(boxes_list, MockMask(masks_xyn), self.names)]

UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)


_plate_detector = None

def get_plate_detector():
    global _plate_detector
    if _plate_detector is None:
        try:
            from ultralytics import YOLO
            from huggingface_hub import hf_hub_download
            # Download and cache the model from Hugging Face
            model_path = hf_hub_download(repo_id="Koushim/yolov8-license-plate-detection", filename="best.pt")
            _plate_detector = YOLO(model_path)
            print("[LPR] YOLOv8-Plate detector model loaded successfully.")
        except Exception as e:
            import traceback
            print(f"[LPR] YOLOv8-Plate detector initialization error: {e}\n{traceback.format_exc()}")
            _plate_detector = False
    return _plate_detector

_easyocr_reader = None

def get_easyocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        try:
            import easyocr
            import torch
            use_gpu = torch.cuda.is_available()
            _easyocr_reader = easyocr.Reader(['en'], gpu=use_gpu)
            print(f"[LPR] EasyOCR Reader loaded successfully (GPU={use_gpu}).")
        except Exception as e:
            import traceback
            print(f"[LPR] EasyOCR initialization error: {e}\n{traceback.format_exc()}")
            _easyocr_reader = False
    return _easyocr_reader

# Vehicle classes that can carry a licence plate
VEHICLE_CLASSES = {'car', 'truck', 'bus', 'motorcycle'}

def _validate_and_format_plate(text: str, conf: float) -> tuple[str, bool]:
    clean = re.sub(r'[^A-Z0-9]', '', text.upper())
    
    if len(clean) < 4 or len(clean) > 12:
        return clean, False

    corrections = {'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '6'}
    
    # Attempt character corrections for India format: AA 00 AA 0000
    if 9 <= len(clean) <= 10:
        corrected = list(clean)
        # 1. State digits (indices 2 and 3) must be digits
        for i in [2, 3]:
            if corrected[i] in corrections:
                corrected[i] = corrections[corrected[i]]
                
        # 2. Middle series letters (index 4, and index 5 if length is 10) must be letters
        num_to_letter = {'8': 'B', '0': 'D', '5': 'S', '1': 'I', '2': 'Z', 'A': 'H'} # Corrects common digit-to-letter OCR confusions (e.g. 8->B, A->H)
        series_indices = [4, 5] if len(clean) == 10 else [4]
        for i in series_indices:
            if corrected[i] in num_to_letter:
                corrected[i] = num_to_letter[corrected[i]]
                
        # 3. Last 4 characters must be digits
        for i in range(len(clean)-4, len(clean)):
            if corrected[i] in corrections:
                corrected[i] = corrections[corrected[i]]
                
        clean_india = "".join(corrected)
        if re.match(r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$', clean_india):
            return clean_india, True

    patterns = [
        r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$', # India
        r'^[0-9]{1,4}[A-Z]{1,3}[0-9]{1,4}$', # Gulf
        r'^[A-Z]{1,3}[0-9]{1,4}[A-Z]{0,3}$', # EU-style
        r'^[A-Z0-9]{4,9}$' # Fallback
    ]
    
    is_valid = any(re.match(p, clean) for p in patterns)
    return clean, is_valid

def try_read_plate(crop_img) -> tuple | None:
    """
    Run OCR on a cropped vehicle region. Uses YOLOv8-Plate to locate and crop
    the exact plate rectangle first, then runs EasyOCR for characters.
    """
    detector = get_plate_detector()
    easy_reader = get_easyocr_reader()
    
    with open("plate_debug.log", "a") as f:
        f.write(f"detector ready: {detector is not None}, easyocr ready: {easy_reader is not None}\n")
        
    try:
        final_crop = crop_img
        plate_box_detected = False
        
        # 1. Use YOLOv8-Plate to find the exact plate boundary inside the vehicle crop
        if detector:
            try:
                det_results = detector(crop_img, verbose=False)
                if det_results and len(det_results[0].boxes) > 0:
                    best_box = max(det_results[0].boxes, key=lambda b: float(b.conf[0]))
                    if float(best_box.conf[0]) > 0.4:
                        px1, py1, px2, py2 = best_box.xyxy[0].tolist()
                        px1, py1, px2, py2 = int(px1), int(py1), int(px2), int(py2)
                        
                        h_c, w_c = crop_img.shape[:2]
                        # 18px horizontal / 6px vertical padding (or proportional to plate size) to avoid character clipping on wide EU/UK/Indian plates
                        plate_w = px2 - px1
                        plate_h = py2 - py1
                        pad_x = max(18, int(plate_w * 0.15))
                        pad_y = max(6, int(plate_h * 0.10))
                        
                        px1 = max(0, px1 - pad_x)
                        py1 = max(0, py1 - pad_y)
                        px2 = min(w_c, px2 + pad_x)
                        py2 = min(h_c, py2 + pad_y)
                        
                        
                        if (px2 - px1) > 10 and (py2 - py1) > 10:
                            final_crop = crop_img[py1:py2, px1:px2]
                            plate_box_detected = True
                            with open("plate_debug.log", "a") as f:
                                f.write(f"YOLO-Plate successful. Box: [{px1}, {py1}, {px2}, {py2}] conf: {float(best_box.conf[0]):.2%}\n")
            except Exception as de:
                with open("plate_debug.log", "a") as f:
                    f.write(f"YOLO-Plate detection exception: {de}\n")

        # If a detector was run but NO plate box was detected, we abort early to save performance
        # and prevent garbage text OCR false-positives on the whole vehicle body.
        if detector and not plate_box_detected:
            return None

        # 2. Run EasyOCR (Primary / Double-Pass System)
        if easy_reader:
            try:
                # Upscale the cropped plate to a robust resolution (min 450px width) for clear character recognition
                h_p, w_p = final_crop.shape[:2]
                if w_p < 450:
                    scale = 450.0 / w_p
                    final_crop = cv2.resize(final_crop, (int(w_p * scale), int(h_p * scale)), interpolation=cv2.INTER_CUBIC)
                
                # Helper to evaluate OCR results list
                def evaluate_ocr(results_list):
                    if not results_list:
                        return None
                    
                    combined_text = "".join([res[1] for res in results_list])
                    clean_combined, is_valid_combined = _validate_and_format_plate(combined_text, 0.8)
                    
                    if is_valid_combined:
                        return (clean_combined, 0.8)
                        
                    best_text = ""
                    best_conf = 0.0
                    for bbox, text, conf in results_list:
                        clean, is_valid = _validate_and_format_plate(text, conf)
                        if is_valid and conf > best_conf:
                            best_text = clean
                            best_conf = conf
                            
                    if best_text and best_conf > 0.4:
                        return (best_text, best_conf)
                    return None

                # --- PASS 1: Bilateral Denoised Crop (Best for preserving sub-pixel font shapes like H vs A, B vs 8) ---
                denoised_crop = cv2.bilateralFilter(final_crop, d=9, sigmaColor=75, sigmaSpace=75)
                ocr_results = easy_reader.readtext(denoised_crop, text_threshold=0.5, link_threshold=0.4)
                
                pass1_res = evaluate_ocr(ocr_results)
                if pass1_res:
                    with open("plate_debug.log", "a") as f:
                        f.write(f"EasyOCR Pass 1 (Bilateral) success: {pass1_res[0]} (conf: {pass1_res[1]:.2%}), YOLO-Local: {plate_box_detected}\n")
                    return pass1_res
                
                # --- PASS 2: CLAHE High Contrast Fallback (Best for extremely faint or low-contrast plates under shadows) ---
                gray_p = cv2.cvtColor(final_crop, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
                gray_p = clahe.apply(gray_p)
                clahe_crop = cv2.cvtColor(gray_p, cv2.COLOR_GRAY2BGR)
                
                ocr_results_clahe = easy_reader.readtext(clahe_crop, text_threshold=0.5, link_threshold=0.4)
                pass2_res = evaluate_ocr(ocr_results_clahe)
                
                if pass2_res:
                    with open("plate_debug.log", "a") as f:
                        f.write(f"EasyOCR Pass 2 (CLAHE) success: {pass2_res[0]} (conf: {pass2_res[1]:.2%}), YOLO-Local: {plate_box_detected}\n")
                    return pass2_res
            except Exception as ee:
                with open("plate_debug.log", "a") as f:
                    f.write(f"EasyOCR reader exception: {ee}\n")

        return None
    except Exception as e:
        print(f"[LPR] try_read_plate error: {e}")
        with open("plate_debug.log", "a") as f:
            f.write(f"Error: {e}\n")
        return None


# ── Main processing task ──────────────────────────────────────────────────────

def process_video_task(video_id: UUID, filepath: str):
    temp_filepath = None
    start_time = time.time()
    db: Session = SessionLocal()
    try:
        # Fetch active target classes
        active_targets = db.query(models.TargetClass).filter(models.TargetClass.is_enabled == True).all()
        target_classes = [t.name.lower() for t in active_targets]

        video = db.query(models.Video).filter(models.Video.id == video_id).first()
        if not video:
            return
        
        video.status = "processing"
        db.commit()

        operator_model = 'yolov8n.pt'
        min_conf = 0.45
        if video.operator_id:
            operator = db.query(models.Operator).filter(models.Operator.id == video.operator_id).first()
            if operator:
                if operator.ai_model:
                    operator_model = operator.ai_model
                if operator.min_confidence is not None:
                    min_conf = operator.min_confidence
        
        # Check processing flags
        process_semantic = getattr(video, 'process_semantic', True)
        process_tags = getattr(video, 'process_tags', True)
        process_ocr = getattr(video, 'process_ocr', True)

        # Check model type to initialize correctly
        model = None
        names = {}
        if process_tags or process_ocr:
            if 'rfdetr' in operator_model.lower():
                model = RFDETRWrapper(operator_model)
            elif 'rtdetr' in operator_model.lower():
                from ultralytics import RTDETR
                model = RTDETR(operator_model)
            else:
                model = YOLO(operator_model)
            names = model.names

        if filepath.startswith("http://") or filepath.startswith("https://"):
            # It is a remote Vercel Blob URL!
            # Download to a temporary file locally for OpenCV processing
            import tempfile
            import requests
            try:
                print(f"[video_processing] Downloading remote video from {filepath} for analysis...")
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
                    response = requests.get(filepath, stream=True)
                    response.raise_for_status()
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            temp_file.write(chunk)
                    temp_filepath = temp_file.name
                print(f"[video_processing] Downloaded to temporary file: {temp_filepath}")
                local_filepath = temp_filepath
            except Exception as e:
                print(f"[video_processing] Failed to download remote video: {e}")
                # fallback to opening URL directly in cv2
                local_filepath = filepath
        else:
            local_filepath = filepath

        cap = cv2.VideoCapture(local_filepath)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        video.duration = duration
        db.commit()

        # 1 FPS sampling (CPU-optimised — halves frame count vs 2 FPS)
        frame_interval = max(1, int(fps))

        current_frame = 0
        last_reported_pct = -1
        processing_progress[video_id] = 0
        frames_since_commit = 0

        is_cancelled = False

        while cap.isOpened():
            if video_id in cancelled_tasks:
                is_cancelled = True
                break

            ret, frame = cap.read()
            if not ret:
                break

            
            if current_frame % frame_interval == 0:
                timestamp_sec = current_frame / fps
                
                # Update live progress
                if total_frames > 0:
                    new_pct = int((current_frame / total_frames) * 100)
                    if new_pct > last_reported_pct:
                        processing_progress[video_id] = new_pct
                        last_reported_pct = new_pct
                
                # Cap at 1280px wide if source resolution is very high
                h_f, w_f = frame.shape[:2]
                if w_f > 1280:
                    scale = 1280.0 / w_f
                    resized_frame = cv2.resize(frame, (int(w_f * scale), int(h_f * scale)))
                else:
                    resized_frame = frame.copy()
                
                # --- BLIP Frame Embedding & Captioning ---
                try:
                    with open("debug_blip.log", "a") as dbg:
                        dbg.write(f"Frame {current_frame} processing... vector_service is {vector_service is not None}\n")
                except:
                    pass
                if process_semantic and vector_service:
                    try:
                        # Convert CV2 BGR to RGB
                        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(rgb_frame)

                        emb = vector_service.get_image_embedding(pil_img)
                        try:
                            caption = vector_service.generate_caption(pil_img)
                        except Exception as ce:
                            caption = ""

                        frame_emb = models.FrameEmbedding(
                            video_id=video_id,
                            timestamp_sec=timestamp_sec,
                            embedding=emb,
                            caption=caption
                        )
                        db.add(frame_emb)
                    except Exception as ve:
                        print(f"[BLIP] Error generating vector/caption: {ve}")
                # -----------------------------------------
                
                if model is not None:
                    # Use .track() instead of direct inference to enable object tracking
                    results = model.track(resized_frame, persist=True, tracker="bytetrack.yaml", conf=min_conf, verbose=False)
                    
                    for r in results:
                        boxes = r.boxes
                        masks = r.masks.xyn if (r.masks is not None) else None
                        
                        for idx, box in enumerate(boxes):
                            cls_id = int(box.cls[0])
                            class_name = names[cls_id].lower()
                            
                            if class_name in target_classes:
                                conf = float(box.conf[0])
                                xyxy = box.xyxy[0]
                                if hasattr(xyxy, 'tolist'):
                                    xyxy = xyxy.tolist()
                                xyxyn = box.xyxyn[0]
                                if hasattr(xyxyn, 'tolist'):
                                    xyxyn = xyxyn.tolist()
                                x1, y1, x2, y2 = [int(v) for v in xyxy]
    
                                track_id = int(box.id[0]) if box.id is not None else None
                                
                                # Extract segmentation polygon contour if available
                                poly_coords = None
                                if masks is not None and idx < len(masks):
                                    poly_coords = masks[idx]
                                    if hasattr(poly_coords, 'tolist'):
                                        poly_coords = poly_coords.tolist()
                                
                                bbox_width = float(x2 - x1)
                                bbox_height = float(y2 - y1)
                                bbox_center_x = float((x1 + x2) / 2.0)
                                bbox_center_y = float((y1 + y2) / 2.0)
                                
                                det = None
                                if process_tags:
                                    # Save object detection with segmentation outline
                                    det = models.Detection(
                                        video_id=video_id,
                                        timestamp_sec=timestamp_sec,
                                        object_type=class_name,
                                        confidence=conf,
                                        bbox_json=json.dumps(xyxyn),
                                        segmentation_json=json.dumps(poly_coords) if poly_coords else None,
                                        track_id=track_id,
                                        bbox_width=bbox_width,
                                        bbox_height=bbox_height,
                                        bbox_center_x=bbox_center_x,
                                        bbox_center_y=bbox_center_y
                                    )
                                    db.add(det)
    
                                # ── LPR: attempt plate read on vehicle crops ──
                                if process_ocr and class_name in VEHICLE_CLASSES:
                                    vehicle_width = x2 - x1
                                    vehicle_height = y2 - y1
                                    
                                    # Skip OCR on tiny vehicles where plates are unreadable anyway
                                    if vehicle_width > 64 and vehicle_height > 64:
                                        # Smart crop based on vehicle type (expanded vertical range to capture trunk-mounted and grill-mounted plates)
                                        if class_name == 'car':
                                            y_start_pct, y_end_pct = 0.15, 1.0  # Captures middle trunk lid down to bumper
                                        elif class_name == 'truck':
                                            y_start_pct, y_end_pct = 0.20, 1.0
                                        elif class_name == 'bus':
                                            y_start_pct, y_end_pct = 0.20, 1.0
                                        elif class_name == 'motorcycle':
                                            y_start_pct, y_end_pct = 0.10, 0.95
                                        else:
                                            y_start_pct, y_end_pct = 0.15, 1.0

                                        plate_y1 = y1 + int((y2 - y1) * y_start_pct)
                                        plate_y2 = y1 + int((y2 - y1) * y_end_pct)
                                        
                                        # 12% horizontal margin to prevent any character clipping on wide vehicle bodies
                                        margin_x = int(vehicle_width * 0.12)
                                        plate_x1 = max(0, x1 - margin_x)
                                        plate_x2 = min(resized_frame.shape[1], x2 + margin_x)
                                        
                                        crop = resized_frame[plate_y1:plate_y2, plate_x1:plate_x2]
                                        
                                        if crop.size > 0:
                                            plate_result = try_read_plate(crop)
                                            if plate_result:
                                                plate_text, plate_conf = plate_result
                                                h_res, w_res = resized_frame.shape[:2]
                                                norm_plate_box = [
                                                    plate_x1 / w_res,
                                                    plate_y1 / h_res,
                                                    plate_x2 / w_res,
                                                    plate_y2 / h_res
                                                ]
                                                lp = models.LicencePlate(
                                                    video_id=video_id,
                                                    detection_id=det.id if det else None,
                                                    timestamp_sec=timestamp_sec,
                                                    plate_number=plate_text,
                                                    confidence=float(plate_conf),
                                                    bbox_json=json.dumps(norm_plate_box)
                                                )
                                                db.add(lp)
                                                print(f"[LPR] Plate detected: {plate_text} ({plate_conf:.0%}) @ {timestamp_sec:.1f}s")
                            
            current_frame += 1

            # Batch commit every 10 processed frames to reduce DB round-trips
            if current_frame % frame_interval == 0:
                frames_since_commit += 1
                if frames_since_commit >= 10:
                    db.commit()
                    frames_since_commit = 0

        cap.release()
        
        if is_cancelled:
            print(f"[LPR] Video {video_id} processing was cancelled by operator.")
            video.status = "cancelled"
            video.processing_time_sec = time.time() - start_time
            db.commit()
            return
        
        # Mark as 100% done
        processing_progress[video_id] = 100
        
        print(f"[LPR] Finished processing video {video_id}.")
            
        video.status = "completed"
        video.processing_time_sec = time.time() - start_time
        db.commit()

    except Exception as e:
        import traceback
        with open("debug_error.log", "w") as f:
            f.write(traceback.format_exc())
            
        print(f"Error processing video {video_id}: {e}")
        try:
            db.rollback()
            video = db.query(models.Video).filter(models.Video.id == video_id).first()
            if video:
                video.status = "failed"
                video.processing_time_sec = time.time() - start_time
                db.commit()
        except:
            pass
    finally:
        if video_id in cancelled_tasks:
            cancelled_tasks.discard(video_id)
        db.close()
        # Clean up temporary file if it was created
        if temp_filepath and os.path.exists(temp_filepath):
            try:
                os.remove(temp_filepath)
                print(f"[video_processing] Cleaned up temporary file: {temp_filepath}")
            except Exception as e:
                print(f"[video_processing] Failed to delete temporary file {temp_filepath}: {e}")

