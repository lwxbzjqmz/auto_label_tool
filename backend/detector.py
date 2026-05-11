from config import DEFAULT_DETECTOR
from yolov3_mbv2.detector import get_boxes_by_yolov3_mbv2
from grounding_dino.detector import get_boxes_by_large_model

def get_boxes(image_path,detector=None,model_select=None,grounding_dino_options=None):
    '''自动模型选择'''
    if detector is None or detector=="" or detector=="high":
        detector=DEFAULT_DETECTOR

    if detector=="yolo":
        return get_boxes_by_yolov3_mbv2(
            image_path,
            model_select=model_select
        )
    
    if detector=="grounding_dino":
        return get_boxes_by_large_model(
            image_path,
            options=grounding_dino_options
                                        )
    
    raise ValueError(f"Unknown detector:{detector}")