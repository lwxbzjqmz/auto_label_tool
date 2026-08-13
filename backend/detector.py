from config import DEFAULT_DETECTOR
from yolov3_mbv2.detector import get_boxes_by_yolov3_mbv2
from grounding_dino.detector import detect_by_large_model


def get_detect_result(image_path,detector=None,model_select=None,grounding_dino_options=None):
    '''自动模型选择,返回完整检测结果: class_boxes + objects'''
    if detector is None or detector=="" or detector=="default":
        detector=DEFAULT_DETECTOR

    if detector=="yolo":
        class_boxes=get_boxes_by_yolov3_mbv2(
            image_path,
            model_select=model_select
        )

        objects=[]

        for class_box in class_boxes:
            objects.append({
                "name":class_box[0],
                "class_box":class_box,
                "score":None
            })

        return {
            "class_boxes":class_boxes,
            "objects":objects
        }
    
    if detector=="grounding_dino":
        result=detect_by_large_model(
            image_path,
            options=grounding_dino_options
        )

        class_boxes=result.get("class_boxes",[])
        objects=result.get("objects",[])

        if not class_boxes and objects:
            class_boxes=[]
            for obj in objects:
                box=obj.get("box",[])
                if len(box)!=4:
                    continue
                class_boxes.append([
                    obj.get("name") or "object",
                    *box
                ])

        return {
            "class_boxes":class_boxes,
            "objects":objects
        }
    
    raise ValueError(f"Unknown detector:{detector}")
