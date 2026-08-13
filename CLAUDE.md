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
  → grounding_dino: grounding_dino/client.py → HTTP POST to worker on port 15051
       → worker.py (separate process, PyTorch) loads grounding-dino-base model
       → returns boxes + object labels
  → Response: {boxes, objects, box_count}
  → Frontend renders boxes on canvas
  → User edits, then POST /api/save → xml_writer.py writes VOC XML
```

### Request flow for classification

```
Frontend (classify button / shortcut)
  → POST /api/classify {image_name, class_boxes, classifier, model_select, rules?}
  → classify.py dispatches:
       pure_color → pure_color_classify/classify.py (RGB rules on box center crop)
       model      → model_classify/classify.py (tflite, 65×65 input, 0-255 raw pixels)
  → Response: {class_boxes, box_count}
  → Frontend updates box class names on canvas
```

### Key files

- [backend/config.py](backend/config.py) — all paths and hyperparameters; user must edit `GROUNDING_DINO_HF_HOME` and `GROUNDING_DINO_PYTHON_PATH` for their machine
- [backend/app.py](backend/app.py) — Flask routes: `/api/images`, `/api/image/<name>`, `/api/detect`, `/api/save`, `/api/annotation/<name>`, `/api/classify`, `/api/classify_rules`, `/api/classify_models`, `/api/reject_image`, `/api/backup_annotations`, `/api/yolo_models`, `/api/detectors`, `/api/shortcut_config`, `/api/grounding_dino_defaults`, `/api/health`
- [backend/detector.py](backend/detector.py) — dispatches to YOLO or Grounding DINO based on `detector` param; unifies return format `{boxes, objects}`
- [backend/classify.py](backend/classify.py) — classification dispatcher: `pure_color` (RGB rules) or `model` (tflite); unified entry used by `/api/classify`
- [backend/grounding_dino/worker.py](backend/grounding_dino/worker.py) — standalone HTTP server (port 15051) that loads the grounding-dino-base model once and handles `/detect` POST + `/health` GET; `app.py` spawns this as a child process
- [backend/grounding_dino/client.py](backend/grounding_dino/client.py) — sends HTTP request to the worker from the Flask process
- [backend/grounding_dino/detector.py](backend/grounding_dino/detector.py) — wraps `client.detect_by_grounding_dino_worker` as the detection entry for this detector
- [backend/grounding_dino_manager.py](backend/grounding_dino_manager.py) — spawns/stops the worker subprocess, waits for it to be ready
- [backend/yolov3_mbv2/detector.py](backend/yolov3_mbv2/detector.py) — YOLOv3 inference: loads `.tflite` model, preprocesses image (crop 40,0 → 280,240 → resize 144×144), runs inference, postprocesses (decode anchors, NMS, scale coords back to original 320×240)
- [backend/yolov3_mbv2/model_paths.py](backend/yolov3_mbv2/model_paths.py) — resolves YOLO model paths by variant number (`models/yolov3_mbv2/<N>/`) and lists available models
- [backend/pure_color_classify/classify.py](backend/pure_color_classify/classify.py) — RGB-rule color classification over the center crop of each box
- [backend/model_classify/classify.py](backend/model_classify/classify.py) — tflite model classification (65×65 input, 5 classes, expects 0-255 raw pixels; model has its own Rescaling layer)
- [backend/model_classify/model_paths.py](backend/model_classify/model_paths.py) — resolves classification model paths by number (`models/classify/<N>.tflite`) and lists available models
- [backend/xml_writer.py](backend/xml_writer.py) — boxes → Pascal VOC XML (normalizes coords, filters invalid boxes, writes to `data/Annotations/`)
- [backend/xml_reader.py](backend/xml_reader.py) — VOC XML → boxes (parses existing annotations back)
- [backend/reject_manager.py](backend/reject_manager.py) — moves bad images + their XML to `data/Rejected/`
- [backend/backup_manager.py](backend/backup_manager.py) — zips all XML files in `data/Annotations/` to `backups/annotations/`
- [backend/image_manager.py](backend/image_manager.py) — lists `.jpg` files in `data/JPEGImages/`
- [backend/batch_detector_and_save.py](backend/batch_detector_and_save.py) — standalone script for batch detection over an image range and saving XML; not part of the web UI flow
- [backend/batch_model_classify_and_save.py](backend/batch_model_classify_and_save.py) — standalone script for batch model classification of existing annotations and saving XML; not part of the web UI flow
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
│   ├── 2/           # model variant 2 (OpenMV deployment)
│   └── 3/ ... 7/    # more variants; each dir contains:
│       ├── yolo3_iou_smartcar_final.tflite
│       └── yolo3_anchors.txt
│       (6/ and 7/ also contain yolo3_iou_smartcar_final_with_post_processing.tflite)
├── classify/
│   ├── 1.tflite ... 5.tflite   # classification models, selectable by number from the UI
│   └── 2026_06_23_13_23_45/    # training artifacts (.h5 weights, hist.csv, acc/loss curves)
└── large_model/
    └── responses/   # legacy large-model JSON schema (deprecated)
```

### YOLO preprocessing details

The YOLOv3-mbv2 model was trained with a specific crop: the original 320×240 image is cropped to region (40, 0) to (280, 240), then resized to 144×144 for model input. After inference, box coordinates are scaled back to the original 320×240 image space. **Do not change these crop/resize parameters** (`YOLOV3_MBV2_CROP_*`, `YOLOV3_MBV2_MODEL_INPUT_*`, `YOLOV3_MBV2_ORIGINAL_IMAGE_*`) without retraining the model.

### Classification model conventions

The classification models (`models/classify/N.tflite`, input 65×65×3, 5 outputs in order ball/redbag/bluebag/brownbear/whitebear) include their own `Rescaling(scale=1/127.5, offset=-1)` layer. `model_classify/classify.py` therefore feeds **raw 0-255 pixels** — do NOT divide by 255 again in the caller (double normalization makes every input near-black and the model returns one constant class). Keep this convention when converting new models from `.h5` (training script: `vision_module/classify/rsmodel.py`).

## Development workflow

- Work from the repository root. The backend uses repository-relative paths, so starting `backend/app.py` from another directory can select the wrong data or model paths.
- On Windows, activate the `autolabel` environment and run `python backend/app.py`. Keep the terminal running while using the browser UI.
- The Flask process may start the Grounding DINO worker on port 15051. Before debugging detector failures, check both `/api/health` and `http://127.0.0.1:15051/health`.
- `backend/run.ps1` and `experiments/start_grounding_dino.ps1` are helper scripts only. Inspect their paths before using them on a different machine.

## Configuration rules

- Machine-specific values belong in `backend/config.py` or local environment configuration. Do not commit personal Conda paths, model cache paths, or absolute dataset paths.
- Keep the `autolabel` and `grounding_dino` environments separate. Changes to one environment must not assume that packages are installed in the other.
- Treat `data/` and `models/` as local assets. They are ignored by Git and must not be removed, reformatted, or regenerated as part of an ordinary code change.
- When adding a detector, preserve the common detector result shape: `{boxes, objects}`. Update backend dispatch, `/api/detectors`, frontend controls, and documentation together.

## Change guidelines

- Preserve existing Flask API routes and response fields unless the change explicitly requires an API change. Check both success and error responses when modifying a route.
- Keep image-space boxes consistent with the 320x240 source image and verify coordinate conversion at the frontend canvas boundary.
- Annotation changes must preserve valid Pascal VOC XML and must not overwrite unrelated annotations. Use the existing XML reader/writer and manager modules.
- Keep frontend changes in the existing vanilla HTML/CSS/JavaScript structure. Do not introduce a framework or build step without updating the workflow and documentation.
- Avoid broad cleanup of generated files, datasets, model files, or unrelated user changes. Make focused edits that are easy to review.

## Verification checklist

There is currently no automated test, lint, or build command. After backend changes, perform the smallest relevant manual checks:

1. Start the server and request `/api/health`.
2. Request `/api/images` and open one image in the UI.
3. Run the affected detector and check that boxes and labels render in the expected image coordinates.
4. Save one annotation, reload it through `/api/annotation/<name>`, and verify the XML under `data/Annotations/`.
5. For rejection or backup changes, verify the resulting files in their destination directories.

For frontend-only changes, reload the page and exercise image selection, box creation, box editing/deletion, detector invocation, and save. A successful Flask start alone is not sufficient validation of the browser workflow.
