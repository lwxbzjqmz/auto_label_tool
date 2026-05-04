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