from large_model.grounding_dino.client import get_boxes_by_grounding_dino_worker


def get_boxes_by_large_model(image_path):
    '''使用Grounding DINO worker进行自动标注'''
    boxes=get_boxes_by_grounding_dino_worker(image_path)

    return boxes
# from pathlib import Path

# from config import LARGE_MODEL_DEFAULT_RESPONSE_PATH
# from large_model.response_parser import load_large_model_json,parse_large_model_response


# _large_model_cache=None


# def get_boxes_by_large_model(image_path):
#     '''使用large_model的json结果返回当前图片boxes'''
#     image_path=Path(image_path)
#     image_name=image_path.name

#     image_boxes_map=_load_large_model_boxes_map()

#     boxes=image_boxes_map.get(image_name,[])

#     return boxes


# def _load_large_model_boxes_map():
#     '''读取large_model json并缓存'''
#     global _large_model_cache

#     if _large_model_cache is not None:
#         return _large_model_cache

#     data=load_large_model_json(LARGE_MODEL_DEFAULT_RESPONSE_PATH)
#     image_boxes_map=parse_large_model_response(data)

#     _large_model_cache=image_boxes_map

#     return _large_model_cache


# def clear_large_model_cache():
#     '''清空large_model缓存,方便后面json更新后重新读取'''
#     global _large_model_cache
#     _large_model_cache=None#更新json后需要重启后端