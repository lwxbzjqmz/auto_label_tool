from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from config import CLASSIFY_DEFAULT_CLASS_NAME
from model_classify.model_paths import get_classify_model_path

_LABELS = [
    'tennis_ball',
    'red_sandbag',
    'light_blue_sandbag',
    'brown_teddy_bear',
    'white_teddy_bear',
]
_SCORE_THRESHOLD = 0.6

_interpreter_cache = {}
_input_details_cache = {}
_output_details_cache = {}


def classify_class_boxes(image_path,class_boxes,model_select=None,**kwargs):
    '''使用 tflite 分类模型更新 class_boxes 的类别名'''
    if not class_boxes:
        return []

    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f'Image not found: {image_path}')

    interpreter,input_details,output_details = _get_interpreter(model_select)
    input_height,input_width = _get_input_size(input_details[0])

    next_class_boxes = []

    with Image.open(image_path) as img:
        image = img.convert('RGB')
        for class_box in class_boxes:
            next_class_box = _classify_class_box(
                image,
                class_box,
                interpreter,
                input_details[0],
                output_details[0],
                input_width,
                input_height,
            )
            if next_class_box is None:
                continue
            next_class_boxes.append(next_class_box)

    return next_class_boxes


def _classify_class_box(image,class_box,interpreter,input_detail,output_detail,input_width,input_height):
    if not isinstance(class_box,(list,tuple)) or len(class_box) != 5:
        return None

    _,xmin,ymin,xmax,ymax = class_box

    try:
        crop_box = _normalize_box(xmin,ymin,xmax,ymax,image.width,image.height)
    except (TypeError,ValueError):
        return None

    if crop_box is None:
        return None

    crop = image.crop(crop_box)
    input_tensor = _prepare_input_tensor(crop,input_detail,input_width,input_height)

    interpreter.set_tensor(input_detail['index'],input_tensor)
    interpreter.invoke()

    output_tensor = interpreter.get_tensor(output_detail['index'])
    class_name = _decode_class_name(output_tensor,output_detail)

    return [
        class_name,
        int(round(float(xmin))),
        int(round(float(ymin))),
        int(round(float(xmax))),
        int(round(float(ymax))),
    ]


def _get_interpreter(model_select=None):
    '''加载 tflite interpreter'''
    cache_key = str(model_select) if model_select is not None and model_select != '' else 'default'

    if cache_key in _interpreter_cache:
        return (
            _interpreter_cache[cache_key],
            _input_details_cache[cache_key],
            _output_details_cache[cache_key],
        )

    model_path = get_classify_model_path(model_select)
    if not model_path.exists():
        raise FileNotFoundError(f'Model not found: {model_path}')

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    _interpreter_cache[cache_key] = interpreter
    _input_details_cache[cache_key] = input_details
    _output_details_cache[cache_key] = output_details

    return interpreter,input_details,output_details


def _get_input_size(input_detail):
    shape = input_detail.get('shape')
    if shape is None or len(shape) != 4:
        raise ValueError(f'Invalid model input shape:{shape}')
    return int(shape[1]), int(shape[2])


def _normalize_box(xmin,ymin,xmax,ymax,image_width,image_height):
    xmin = int(round(float(xmin)))
    ymin = int(round(float(ymin)))
    xmax = int(round(float(xmax)))
    ymax = int(round(float(ymax)))

    left = max(0,min(xmin,xmax))
    right = min(image_width,max(xmin,xmax))
    top = max(0,min(ymin,ymax))
    bottom = min(image_height,max(ymin,ymax))

    if right - left < 1 or bottom - top < 1:
        return None

    return (left,top,right,bottom)


def _prepare_input_tensor(image,input_detail,input_width,input_height):
    resized = image.resize((input_width,input_height))
    array = np.asarray(resized)

    dtype = input_detail['dtype']
    if dtype == np.float32:
        # 模型内部自带 Rescaling(1/255) 归一化层,这里直接喂 0-255 原始像素
        # (与 OpenMV tf.classify 行为一致;若再除 255 会造成双重归一化,输出恒为同一类)
        array = array.astype(np.float32)
    else:
        array = array.astype(dtype)

    return np.expand_dims(array,axis=0)


def _decode_class_name(output_tensor,output_detail):
    scores = np.asarray(output_tensor).reshape(-1).astype(np.float32)
    if scores.size == 0:
        return CLASSIFY_DEFAULT_CLASS_NAME

    scale,zero_point = output_detail.get('quantization',(0,0))
    if scale:
        scores = (scores - float(zero_point)) * float(scale)

    best_index = int(np.argmax(scores))
    best_score = float(scores[best_index])

    if best_score < _SCORE_THRESHOLD or best_index >= len(_LABELS):
        return CLASSIFY_DEFAULT_CLASS_NAME

    return _LABELS[best_index]
