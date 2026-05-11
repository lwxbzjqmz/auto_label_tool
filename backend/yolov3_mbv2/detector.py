from pathlib import Path
import tensorflow as tf

from config import(
    YOLOV3_MBV2_SCORE_THRESHOLD,
    YOLOV3_MBV2_NMS_IOU_THRESHOLD,
)

from yolov3_mbv2.model_paths import get_yolov3_mbv2_model_path
from yolov3_mbv2.anchors import load_anchors
from yolov3_mbv2.preprocess import load_and_crop_image,prepare_input_tensor
from yolov3_mbv2.postprocess import decode_yolo_output,nms,map_boxes_to_original_image

_interpreter_cache={}
_input_details_cache={}
_output_details_cache={}

def get_boxes_by_yolov3_mbv2(image_path,model_select=None):
    '''调用自己的yolov3_mbv2模型'''
    image_path=Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    model_path=get_yolov3_mbv2_model_path(model_select)

    if not model_path.exists():
        raise FileNotFoundError(
            f"YOLOv3-mbv2 model not found: {model_path}"
        )
    
    interpreter,input_details,output_details=_get_interpreter(model_select)
    anchors=load_anchors(model_select)

    crop_img=load_and_crop_image(image_path)

    input_tensor=prepare_input_tensor(crop_img,input_details[0])

    interpreter.set_tensor(input_details[0]["index"],input_tensor)
    interpreter.invoke()

    raw_output=interpreter.get_tensor(output_details[0]["index"])

    boxes_with_scores=decode_yolo_output(
        raw_output=raw_output,
        output_detail=output_details[0],
        anchors=anchors,
        score_threshold=YOLOV3_MBV2_SCORE_THRESHOLD,
    )

    boxes_with_scores=nms(
        boxes_with_scores,
        iou_threshold=YOLOV3_MBV2_NMS_IOU_THRESHOLD,
    )

    boxes=map_boxes_to_original_image(boxes_with_scores)

    return boxes

def _get_interpreter(model_select=None):
    '''加载 tflite interpreter'''
    cache_key=str(model_select) if model_select is not None and model_select!="" else "default"

    if cache_key in _interpreter_cache:
        return (
            _interpreter_cache[cache_key],
            _input_details_cache[cache_key],
            _output_details_cache[cache_key]
        )
    
    model_path=get_yolov3_mbv2_model_path(model_select)

    interpreter=tf.lite.Interpreter(
        model_path=str(model_path)
    )

    interpreter.allocate_tensors()

    input_details=interpreter.get_input_details()
    output_details=interpreter.get_output_details()

    _interpreter_cache[cache_key]=interpreter
    _input_details_cache[cache_key]=input_details
    _output_details_cache[cache_key]=output_details

    return interpreter,input_details,output_details
# from pathlib import Path
# import tensorflow as tf

# from config import(
#     YOLOV3_MBV2_PC_MODEL_PATH,
#     YOLOV3_MBV2_SCORE_THRESHOLD,
#     YOLOV3_MBV2_NMS_IOU_THRESHOLD,
# )

# from yolov3_mbv2.anchors import load_anchors
# from yolov3_mbv2.preprocess import load_and_crop_image,prepare_input_tensor
# from yolov3_mbv2.postprocess import decode_yolo_output,nms,map_boxes_to_original_image

# _interpreter=None
# _input_details=None
# _output_details=None

# def get_boxes_by_yolov3_mbv2(image_path):
#     '''调用自己的yolov3_mbv2模型'''
#     image_path=Path(image_path)

#     if not image_path.exists():
#         raise FileNotFoundError(f"Image not found: {image_path}")
    
#     if not YOLOV3_MBV2_PC_MODEL_PATH.exists():
#         raise FileNotFoundError(
#             f"YOLOv3-mbv2 model not found: {YOLOV3_MBV2_PC_MODEL_PATH}"
#         )
    
#     interpreter,input_details,output_details=_get_interpreter()
#     anchors=load_anchors()

#     crop_img=load_and_crop_image(image_path)

#     input_tensor=prepare_input_tensor(crop_img,input_details[0])

#     interpreter.set_tensor(input_details[0]["index"],input_tensor)
#     interpreter.invoke()

#     raw_output=interpreter.get_tensor(output_details[0]["index"])

#     boxes_with_scores=decode_yolo_output(
#         raw_output=raw_output,
#         output_detail=output_details[0],
#         anchors=anchors,
#         score_threshold=YOLOV3_MBV2_SCORE_THRESHOLD,
#     )

#     boxes_with_scores=nms(
#         boxes_with_scores,
#         iou_threshold=YOLOV3_MBV2_NMS_IOU_THRESHOLD,
#     )

#     boxes=map_boxes_to_original_image(boxes_with_scores)

#     return boxes

# def _get_interpreter():
#     '''加载 tflite interpreter'''
#     global _interpreter,_input_details,_output_details

#     if _interpreter is not None:
#         return _interpreter,_input_details,_output_details
    
#     _interpreter=tf.lite.Interpreter(
#         model_path=str(YOLOV3_MBV2_PC_MODEL_PATH)
#     )

#     _interpreter.allocate_tensors()

#     _input_details=_interpreter.get_input_details()
#     _output_details=_interpreter.get_output_details()

#     return _interpreter,_input_details,_output_details