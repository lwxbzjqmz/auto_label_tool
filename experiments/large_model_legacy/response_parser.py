import json
from pathlib import Path 
from large_model.schema import(
    IMAGE_NAME_KEY,
    OBJECTS_KEY,
    BOX_KEY,
)

def load_large_model_json(json_path):
    '''读取large_model返回的json文件'''
    json_path=Path(json_path)

    if not json_path.exists():
        raise FileNotFoundError(f"large_model json not found:{json_path}")
    
    text=json_path.read_text(encoding="utf-8")

    try:
        data=json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"large_model json解析失败:{json_path},{e}")
    
    return data

def parse_large_model_response(data):
    '''解析large_model返回结果
        {
        "00000.jpg":[
            [xmin,ymin,xmax,ymax],
            [xmin,ymin,xmax,ymax]
        ],
        "00001.jpg":[]
    }
    '''

    if not isinstance(data,dict):
        raise ValueError("large_model返回的结果必须是dict")
    
    images=data.get("images")

    if not isinstance(images,list):
        raise ValueError("large_model返回结果缺少images列表")
    
    result={}

    for image_item in images:
        if not isinstance(image_item,dict):
            continue

        image_name=image_item.get(IMAGE_NAME_KEY)

        if not image_name:
            continue

        objects=image_item.get(OBJECTS_KEY,[])

        if not isinstance(objects,list):
            objects=[]

    boxes=[]

    for obj in objects:
        if not isinstance(obj,dict):
            continue

        box=obj.get(BOX_KEY)

        normalized_box=_normalize_box(box)

        if normalized_box is not None:
            boxes.append(normalized_box)

    result[image_name]=boxes

    return result

def _normalize_box(box):
    '''清洗box'''
    if not isinstance(box,list):
        return None
    
    if len(box)!=4:
        return None
    
    try:
        xmin=int(round(float(box[0])))
        ymin=int(round(float(box[1])))
        xmax=int(round(float(box[2])))
        ymax=int(round(float(box[3])))
    except (TypeError,ValueError):
        return None
    
    left=min(xmin,xmax)
    right=max(xmin,xmax)
    top=min(ymin,ymax)
    bottom=max(ymin,ymax)

    if right-left<1 or bottom-top<1:
        return None
    
    return [left,top,right,bottom]


    


    
