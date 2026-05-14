from config import DEFAULT_DETECTOR
from yolov3_mbv2.detector import get_boxes_by_yolov3_mbv2
from grounding_dino.detector import detect_by_large_model


def get_detect_result(image_path,detector=None,model_select=None,grounding_dino_options=None):
    '''自动模型选择,返回完整检测结果: boxes + objects'''
    if detector is None or detector=="" or detector=="default":
        detector=DEFAULT_DETECTOR

    if detector=="yolo":
        boxes=get_boxes_by_yolov3_mbv2(
            image_path,
            model_select=model_select
        )

        objects=[]

        for box in boxes:
            objects.append({
                "name":"object",
                "box":box,
                "score":None
            })

        return {
            "boxes":boxes,
            "objects":objects
        }
    
    if detector=="grounding_dino":
        result=detect_by_large_model(
            image_path,
            options=grounding_dino_options
        )

        boxes=result.get("boxes",[])
        objects=result.get("objects",[])

        return {
            "boxes":boxes,
            "objects":objects
        }
    
    raise ValueError(f"Unknown detector:{detector}")
