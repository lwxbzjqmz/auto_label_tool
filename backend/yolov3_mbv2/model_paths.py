from config import(
    YOLOV3_MODEL_SELECT,
    YOLOV3_MBV2_MODEL_ROOT,
    YOLOV3_MBV2_MODEL_FILENAME,
    YOLOV3_MBV2_ANCHORS_FILENAME,
)

def normalize_model_select(model_select=None):
    '''处理yolo模型编号'''
    if model_select is None or model_select=="":
        return YOLOV3_MODEL_SELECT

    return model_select

def _find_tflite_path(model_dir):
    '''在模型目录中查找 tflite 文件,优先使用配置的文件名'''
    preferred = model_dir / YOLOV3_MBV2_MODEL_FILENAME
    if preferred.exists():
        return preferred

    tflite_files = sorted(model_dir.glob("*.tflite"))
    if tflite_files:
        return tflite_files[0]

    return None

def _find_anchors_path(model_dir):
    '''在模型目录中查找 anchors 文件,优先使用配置的文件名'''
    preferred = model_dir / YOLOV3_MBV2_ANCHORS_FILENAME
    if preferred.exists():
        return preferred

    anchors_files = sorted(model_dir.glob("*anchors*.txt"))
    if anchors_files:
        return anchors_files[0]

    return None

def get_yolov3_mbv2_model_path(model_select=None):
    '''获取指定YOLO模型路径'''
    model_select=normalize_model_select(model_select)
    model_dir = YOLOV3_MBV2_MODEL_ROOT / str(model_select)
    path = _find_tflite_path(model_dir)
    if path is not None:
        return path
    return model_dir / YOLOV3_MBV2_MODEL_FILENAME

def get_yolov3_mbv2_anchors_path(model_select=None):
    '''获取指定YOLO anchors路径'''
    model_select=normalize_model_select(model_select)
    model_dir = YOLOV3_MBV2_MODEL_ROOT / str(model_select)
    path = _find_anchors_path(model_dir)
    if path is not None:
        return path
    return model_dir / YOLOV3_MBV2_ANCHORS_FILENAME

def list_yolov3_mbv2_models():
    '''列出可用模型'''
    models=[]

    if not YOLOV3_MBV2_MODEL_ROOT.exists():
        return models

    for item in YOLOV3_MBV2_MODEL_ROOT.iterdir():
        if not item.is_dir():
            continue

        tflite_path = _find_tflite_path(item)
        if tflite_path is None:
            continue

        anchors_path = _find_anchors_path(item)

        models.append({
            "id": item.name,
            "name": f"YOLOv3-mbv2 模型 {item.name}",
            "tflite_file": tflite_path.name,
            "anchors_file": anchors_path.name if anchors_path else None,
            "has_anchors": anchors_path is not None,
        })

    models.sort(key=lambda x: int(x["id"]) if x["id"].isdigit() else x["id"])

    return models

