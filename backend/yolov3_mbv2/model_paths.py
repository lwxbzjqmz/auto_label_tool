from config import(
    MODEL_SELECT,
    YOLOV3_MBV2_MODEL_ROOT,
    YOLOV3_MBV2_MODEL_FILENAME,
    YOLOV3_MBV2_ANCHORS_FILENAME,
)

def normalize_model_select(model_select=None):
    '''处理yolo模型编号'''
    if model_select is None or model_select=="":
        return MODEL_SELECT
    
    return model_select

def get_yolov3_mbv2_model_path(model_select=None):
    '''获取指定YOLO模型路径'''
    model_select=normalize_model_select(model_select)

    return (YOLOV3_MBV2_MODEL_ROOT
            /str(model_select)
            /YOLOV3_MBV2_MODEL_FILENAME)

def get_yolov3_mbv2_anchors_path(model_select=None):
    '''获取指定YOLO anchors路径'''
    model_select=normalize_model_select(model_select)

    return (YOLOV3_MBV2_MODEL_ROOT
            /str(model_select)
            /YOLOV3_MBV2_ANCHORS_FILENAME)

def list_yolov3_mbv2_models():
    '''列出可用模型'''
    models=[]

    if not YOLOV3_MBV2_MODEL_ROOT.exists():
        return models
    
    for item in YOLOV3_MBV2_MODEL_ROOT.iterdir():
        if not item.is_dir():
            continue

        model_path=item/YOLOV3_MBV2_MODEL_FILENAME
        anchors_path=item/YOLOV3_MBV2_ANCHORS_FILENAME

        if model_path.exists() and anchors_path.exists():
            models.append({
                "id":item.name,
                "name":f"YOLOv3-mbv2 模型 {item.name}"
            })

    models.sort(key=lambda x:int(x["id"]) if str(x["id"].isdigit()) else str(x["id"]))

    return models
