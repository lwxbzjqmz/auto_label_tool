# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

A web-based auto-labeling tool for object detection datasets. Users draw/edit bounding boxes on images, run AI detectors (YOLOv3-mbv2 / Grounding DINO), and save annotations as Pascal VOC XML. Built for the "蚂蚁搬家" (ant moving) dataset at Nanjing University of Information Science and Technology.

## Commands

```bash
# Activate the main conda environment, then start the dev server
conda activate autolabel
python backend/app.py
# Opens at http://127.0.0.1:5000
```

No build/lint/test tooling exists in this project.

## Architecture

**Backend**: Flask server with REST API on port 5000. All backend Python lives under `backend/`.

**Frontend**: Vanilla HTML/CSS/JS single-page app (no framework). Canvas-based image display with bounding box drawing/editing.

**Two conda environments** are required — they cannot be merged because YOLO uses TensorFlow and Grounding DINO uses PyTorch:

| Environment | Purpose |
|---|---|
| `autolabel` | Flask server, YOLOv3-mbv2 inference (TensorFlow Lite) |
| `grounding_dino` | Grounding DINO worker process (PyTorch + Transformers), started as a subprocess by `app.py` |

### Request flow for detection

```
Frontend (canvas drawing / button click)
  → POST /api/detect {image_name, detector, ...params}
  → detector.py selects: yolo or grounding_dino
  → yolo: yolov3_mbv2/detector.py → TensorFlow Lite (.tflite) inference
  → grounding_dino: grounding_dino/client.py → HTTP POST to worker on port 5051
       → worker.py (separate process, PyTorch) loads grounding-dino-base model
       → returns boxes + object labels
  → Response: {boxes, objects, box_count}
  → Frontend renders boxes on canvas
  → User edits, then POST /api/save → xml_writer.py writes VOC XML
```

### Key files

- [backend/config.py](backend/config.py) — all paths and hyperparameters; user must edit `GROUNDING_DINO_HF_HOME` and `GROUNDING_DINO_PYTHON_PATH` for their machine
- [backend/app.py](backend/app.py) — Flask routes: `/api/images`, `/api/image/<name>`, `/api/detect`, `/api/save`, `/api/annotation/<name>`, `/api/reject_image`, `/api/backup_annotations`, `/api/yolo_models`, `/api/detectors`, `/api/health`
- [backend/detector.py](backend/detector.py) — dispatches to YOLO or Grounding DINO based on `detector` param; unifies return format `{boxes, objects}`
- [backend/grounding_dino/worker.py](backend/grounding_dino/worker.py) — standalone HTTP server (port 5051) that loads the grounding-dino-base model once and handles `/detect` POST + `/health` GET; `app.py` spawns this as a child process
- [backend/grounding_dino/client.py](backend/grounding_dino/client.py) — sends HTTP request to the worker from the Flask process
- [backend/grounding_dino_manager.py](backend/grounding_dino_manager.py) — spawns/stops the worker subprocess, waits for it to be ready
- [backend/yolov3_mbv2/detector.py](backend/yolov3_mbv2/detector.py) — YOLOv3 inference: loads `.tflite` model, preprocesses image (crop 40,0 → 280,240 → resize 144×144), runs inference, postprocesses (decode anchors, NMS, scale coords back to original 320×240)
- [backend/xml_writer.py](backend/xml_writer.py) — boxes → Pascal VOC XML (normalizes coords, filters invalid boxes, writes to `data/Annotations/`)
- [backend/xml_reader.py](backend/xml_reader.py) — VOC XML → boxes (parses existing annotations back)
- [backend/reject_manager.py](backend/reject_manager.py) — moves bad images + their XML to `data/Rejected/`
- [backend/backup_manager.py](backend/backup_manager.py) — zips all XML files in `data/Annotations/` to `backups/annotations/`
- [backend/image_manager.py](backend/image_manager.py) — lists `.jpg` files in `data/JPEGImages/`
- [backend/batch_runner_and_save.py](backend/batch_runner_and_save.py) — standalone script for batch-detecting a range of images via Grounding DINO and saving XML; not part of the web UI flow
- [frontend/main.js](frontend/main.js) — all frontend logic: image list, canvas rendering, box drawing/selection/deletion, detector calls, save, keyboard shortcuts, preview wall
- [frontend/style.css](frontend/style.css) — all styles
- [frontend/index.html](frontend/index.html) — single-page layout (left panel: image list, center: canvas, right: controls + params)

### Data directory structure

```
data/
├── JPEGImages/      # .jpg source images
├── Annotations/     # generated .xml VOC annotations
└── Rejected/        # rejected images moved here
    ├── JPEGImages/
    └── Annotations/
```

### Models

```
models/
├── yolov3_mbv2/
│   ├── 1/           # model variant 1 (PC)
│   │   ├── yolo3_iou_smartcar_final.tflite
│   │   └── yolo3_anchors.txt
│   └── 2/           # model variant 2 (OpenMV deployment)
│       ├── yolo3_iou_smartcar_final.tflite
│       └── yolo3_anchors.txt
└── large_model/
    └── responses/   # legacy large-model JSON schema (deprecated)
```

### YOLO preprocessing details

The YOLOv3-mbv2 model was trained with a specific crop: the original 320×240 image is cropped to region (40, 0) to (280, 240), then resized to 144×144 for model input. After inference, box coordinates are scaled back to the original 320×240 image space. **Do not change these crop/resize parameters** (`YOLOV3_MBV2_CROP_*`, `YOLOV3_MBV2_MODEL_INPUT_*`, `YOLOV3_MBV2_ORIGINAL_IMAGE_*`) without retraining the model.
