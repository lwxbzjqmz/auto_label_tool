from config import MODEL_SELECT
from yolov3_mbv2.detector import get_boxes_by_yolov3_mbv2
from grounding_dino.detector import get_boxes_by_large_model

def get_boxes(image_path):
    '''自动模型选择'''
    if MODEL_SELECT==0:
        return get_boxes_by_large_model(image_path)
    else:
        return get_boxes_by_yolov3_mbv2(image_path)
    
