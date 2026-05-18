from yolov3_mbv2.model_paths import get_yolov3_mbv2_anchors_path

_anchors_cache={}

def load_anchors(model_select=None):
    '''读取anchors文件,返回元组列表'''
    anchors_path=get_yolov3_mbv2_anchors_path(model_select)
    cache_key=str(model_select) if model_select is not None and model_select!="" else "default"

    if cache_key in _anchors_cache:
        return _anchors_cache[cache_key]
    
    if not anchors_path.exists():
        raise FileNotFoundError(
            f"YOLOv3-mbv2 anchors not found for model {model_select or 'default'}: {anchors_path}"
        )
    
    text=anchors_path.read_text(encoding="utf-8").strip()
    values=[]

    for part in text.replace("\n",",").split(","):
        part=part.strip()
        if part:
            values.append(float(part))

    if len(values)%2!=0:
        raise ValueError(f"anchors 数量不成对: {text}")

    anchors=[]

    for i in range(0,len(values),2):
        anchors.append((values[i],values[i+1]))

    if len(anchors)!=3:
        raise ValueError(f"anchors数据非3组 {len(anchors)} 组: {anchors}")

    _anchors_cache[cache_key]=anchors
    return _anchors_cache[cache_key]
# from yolov3_mbv2.model_paths import get_yolov3_mbv2_anchors_path,normalize_model_select


# anchors_path=get_yolov3_mbv2_anchors_path(model_select)

# _anchors=None

# def load_anchors():
#     '''读取anchors文件,返回元组列表'''
#     global _anchors

#     if _anchors is not None:
#         return _anchors
    
#     if not YOLOV3_MBV2_ANCHORS_PATH.exists():
#         raise FileNotFoundError(
#             f"YOLOv3-mbv2 anchors not found: {YOLOV3_MBV2_ANCHORS_PATH}"
#         )
    
#     text=YOLOV3_MBV2_ANCHORS_PATH.read_text(encoding="utf-8").strip()
#     values=[]

#     for part in text.replace("\n",",").split(","):
#         part=part.strip()
#         if part:
#             values.append(float(part))

#     if len(values)%2!=0:
#         raise ValueError(f"anchors 数量不成对: {text}")

#     anchors=[]

#     for i in range(0,len(values),2):
#         anchors.append((values[i],values[i+1]))

#     if len(anchors)!=3:
#         raise ValueError(f"anchors数据非3组 {len(anchors)} 组: {anchors}")

#     _anchors=anchors
#     return _anchors