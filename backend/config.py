from pathlib import Path

BASE_DIR=Path(__file__).resolve().parent.parent#项目根目录

FRONTEND_DIR=BASE_DIR/"frontend"#前端目录

DATA_DIR=BASE_DIR/"data"#数据集目录

IMAGE_DIR=DATA_DIR/"JPEGImages"#图片目录

ANNOTATION_DIR=DATA_DIR/"Annotations"#XML标注目录

IMAGE_EXTENSIONS={".jpg"}#图片后缀

CLASS_NAME="object"#XML中的类别名

VOC_FOLDER_NAME="detect.class"#XML中，数据集的类别名

MODEL_DIR=BASE_DIR/"models"#模型根目录

MODEL_SELECT=0#模型选择flag，0是large_model路径

YOLOV3_MBV2_PC_MODEL_PATH=(MODEL_DIR
                           /"yolov3_mbv2"
                           /str(MODEL_SELECT)
                           /"yolo3_iou_smartcar_final.tflite")
#模型有两个，一个用于PC端，一个部署于openmv

YOLOV3_MBV2_ANCHORS_PATH=(MODEL_DIR
                           /"yolov3_mbv2"
                           /str(MODEL_SELECT)
                           /"yolo3_anchors.txt")

#========YOLOv3-mbv2的配置===========
YOLOV3_MBV2_SCORE_THRESHOLD=0.6
YOLOV3_MBV2_NMS_IOU_THRESHOLD=0.3

YOLOV3_MBV2_CROP_X=40
YOLOV3_MBV2_CROP_Y=0
YOLOV3_MBV2_CROP_W=240
YOLOV3_MBV2_CROP_H=240

YOLOV3_MBV2_MODEL_INPUT_W=144
YOLOV3_MBV2_MODEL_INPUT_H=144

YOLOV3_MBV2_ORIGINAL_IMAGE_W=320
YOLOV3_MBV2_ORIGINAL_IMAGE_H=240

#========large_model的配置===========
LARGE_MODEL_DIR=MODEL_DIR/"large_model"
LARGE_MODEL_RESPONSES_DIR=LARGE_MODEL_DIR/"responses"
LARGE_MODEL_DEFAULT_RESPONSE_PATH=LARGE_MODEL_RESPONSES_DIR/"large_model_result.json"

# ======= Grounding DINO 本地worker配置 =========

GROUNDING_DINO_WORKER_HOST="127.0.0.1"

GROUNDING_DINO_WORKER_PORT=5051

GROUNDING_DINO_WORKER_URL=f"http://{GROUNDING_DINO_WORKER_HOST}:{GROUNDING_DINO_WORKER_PORT}/detect"

GROUNDING_DINO_MODEL_ID="IDEA-Research/grounding-dino-base"

GROUNDING_DINO_HF_HOME=r"C:\Users\lwx\.cache\huggingface"

GROUNDING_DINO_LOCAL_FILES_ONLY=True

GROUNDING_DINO_DEFAULT_PROMPT="tennis ball. blue bean bag. red bean bag. white teddy bear. brown teddy bear"

GROUNDING_DINO_BOX_THRESHOLD=0.25

GROUNDING_DINO_TEXT_THRESHOLD=0.20

GROUNDING_DINO_REQUEST_TIMEOUT=60

#=====Grounding DINO worker进程配置========
GROUNDING_DINO_AUTO_START=True

GROUNDING_DINO_PYTHON_PATH=r"W:\anaconda3\envs\grounding_dino\python.exe"

GROUNDING_DINO_WORKER_SCRIPT=BASE_DIR/"backend"/"large_model"/"grounding_dino/worker.py"

GROUNDING_DINO_STARTUP_TIMEOUT=120
