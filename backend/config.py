from pathlib import Path

BASE_DIR=Path(__file__).resolve().parent.parent#项目根目录

FRONTEND_DIR=BASE_DIR/"frontend"#前端目录

DATA_DIR=BASE_DIR/"data"#数据集目录

IMAGE_DIR=DATA_DIR/"JPEGImages"#图片目录

ANNOTATION_DIR=DATA_DIR/"Annotations"#XML标注目录

REJECTED_DATA_DIR=DATA_DIR/"Rejected"

REJECTED_IMAGE_DIR=REJECTED_DATA_DIR/"JPEGImages"

REJECTED_ANNOTATION_DIR=REJECTED_DATA_DIR/"Annotations"

IMAGE_EXTENSIONS={".jpg"}#图片后缀

CLASS_NAME=['object', 'red_sandbag', 'light_blue_sandbag', 'tennis_ball', 'white_teddy_bear', 'brown_teddy_bear']#XML中的类别名的列表

VOC_FOLDER_NAME="detect.class"#XML中，数据集的类别名

MODEL_DIR=BASE_DIR/"models"#模型根目录

YOLOV3_MODEL_SELECT=1#YOLOv3模型选择flag，从1开始

DEFAULT_DETECTOR="yolo"

YOLOV3_MBV2_MODEL_ROOT=MODEL_DIR/"yolov3_mbv2"

YOLOV3_MBV2_MODEL_FILENAME="yolo3_iou_smartcar_final.tflite"

YOLOV3_MBV2_ANCHORS_FILENAME="yolo3_anchors.txt"

YOLOV3_MBV2_PC_MODEL_PATH=(YOLOV3_MBV2_MODEL_ROOT
                           /str(YOLOV3_MODEL_SELECT)
                           /YOLOV3_MBV2_MODEL_FILENAME)
#模型目录，每个子目录代表一个模型变体

YOLOV3_MBV2_ANCHORS_PATH=(YOLOV3_MBV2_MODEL_ROOT
                           /str(YOLOV3_MODEL_SELECT)
                           /YOLOV3_MBV2_ANCHORS_FILENAME)

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

GROUNDING_DINO_WORKER_PORT=15051

GROUNDING_DINO_WORKER_URL=f"http://{GROUNDING_DINO_WORKER_HOST}:{GROUNDING_DINO_WORKER_PORT}/detect"

GROUNDING_DINO_MODEL_ID="IDEA-Research/grounding-dino-base"

GROUNDING_DINO_HF_HOME=r"C:\Users\ASUS\.cache\huggingface"

GROUNDING_DINO_LOCAL_FILES_ONLY=True   # False=首次联网下载模型；下载完成缓存后可视需要改回 True 以跳过联网检查

GROUNDING_DINO_DEFAULT_PROMPT="tennis ball . light blue sandbag . red sandbag . white teddy bear . brown teddy bear"

GROUNDING_DINO_BOX_THRESHOLD=0.25

GROUNDING_DINO_TEXT_THRESHOLD=0.20

GROUNDING_DINO_REQUEST_TIMEOUT=60

GROUNDING_DINO_USE_NMS=True

GROUNDING_DINO_NMS_IOU_THRESHOLD=0.20

#=====Grounding DINO worker进程配置========
GROUNDING_DINO_AUTO_START=True

GROUNDING_DINO_PYTHON_PATH=r"C:\Users\ASUS\miniconda3\envs\grounding_dino\python.exe"

GROUNDING_DINO_WORKER_SCRIPT=BASE_DIR/"backend"/"grounding_dino/worker.py"

GROUNDING_DINO_STARTUP_TIMEOUT=120

#========pure_color_classify分类配置==========
CLASSIFY_DEFAULT_CLASS_NAME="object"

CLASSIFY_CENTER_RATIO=0.5#取box中间区域的比例

CLASSIFIER_PURE_COLOR="pure_color"
CLASSIFIER_MODEL="model"

DEFAULT_CLASSIFIER=CLASSIFIER_PURE_COLOR

#rgb均值范围,用于简单颜色分类,后续根据实际图片继续调整
CLASSIFY_PIXEL_RULES=[{'name': 'red_sandbag', 'rgb_max': (255, 105, 105), 'rgb_min': (120, 0, 0)},
 {'name': 'light_blue_sandbag', 'rgb_max': (90, 235, 255), 'rgb_min': (0, 70, 160)},
 {'name': 'tennis_ball', 'rgb_max': (235, 235, 145), 'rgb_min': (90, 155, 0)},
 {'name': 'white_teddy_bear', 'rgb_max': (255, 255, 255), 'rgb_min': (150, 150, 150)},
 {'name': 'brown_teddy_bear', 'rgb_max': (150, 145, 115), 'rgb_min': (34, 15, 5)}]

#========model_classify分类配置==========
CLASSIFY_MODEL_ROOT=MODEL_DIR/"classify"
# 分类模型直接放在根目录,按编号命名(1.tflite, 2.tflite, ...),
# 与 yolov3 一样可在前端下拉选择模型;也兼容子目录(1/xxx.tflite)与旧命名 classify.tflite

CLASSIFY_MODEL_FILENAME="classify.tflite"#子目录内的首选文件名,以及旧命名兼容

CLASSIFY_MODEL_SELECT=1#分类模型选择flag，从1开始

MODEL_CLASSIFY_PATH=CLASSIFY_MODEL_ROOT/CLASSIFY_MODEL_FILENAME#旧命名兼容路径

