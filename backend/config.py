from pathlib import Path

BASE_DIR=Path(__file__).resolve().parent.parent#项目根目录

FRONTEND_DIR=BASE_DIR/"frontend"#前端目录

DATA_DIR=BASE_DIR/"data"#数据集目录

IMAGE_DIR=DATA_DIR/"JPEGImages"#图片目录

ANNOTATION_DIR=DATA_DIR/"Annotations"#XML标注目录

IMAGE_EXTENSIONS={".jpg"}#图片后缀

CLASS_NAME="object"#XML中的类别名

VOC_FOLDER_NAME="detect.class"#XML中，数据集的类别名

