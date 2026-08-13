from PIL import Image,ImageStat
from config import(
    CLASSIFY_DEFAULT_CLASS_NAME,
    CLASSIFY_CENTER_RATIO,
    CLASSIFY_PIXEL_RULES,
)


def classify_class_boxes(image_path,class_boxes,rules=None):
    '''根据box中间区域像素,更新class_boxes中的类别名'''
    if not class_boxes:
        return []

    if rules is None:
        rules=CLASSIFY_PIXEL_RULES

    next_class_boxes=[]

    with Image.open(image_path) as img:
        image=img.convert("RGB")

        for class_box in class_boxes:
            next_class_box=classify_class_box(image,class_box,rules)
            if next_class_box is None:
                continue
            next_class_boxes.append(next_class_box)

    return next_class_boxes


def classify_class_box(image,class_box,rules=None):
    '''分类单个class_box'''
    if len(class_box)!=5:
        return None

    if rules is None:
        rules=CLASSIFY_PIXEL_RULES

    _,xmin,ymin,xmax,ymax=class_box

    try:
        crop_box=_get_center_crop_box(
            xmin,
            ymin,
            xmax,
            ymax,
            image.width,
            image.height
        )
    except (TypeError,ValueError):
        return None

    if crop_box is None:
        return None

    crop=image.crop(crop_box)
    rgb_mean=_get_rgb_mean(crop)
    class_name=_match_class_name(rgb_mean,rules)

    return [
        class_name,
        int(round(float(xmin))),
        int(round(float(ymin))),
        int(round(float(xmax))),
        int(round(float(ymax))),
    ]


def _get_center_crop_box(xmin,ymin,xmax,ymax,image_width,image_height):
    '''获取box中间区域'''
    xmin=int(round(float(xmin)))
    ymin=int(round(float(ymin)))
    xmax=int(round(float(xmax)))
    ymax=int(round(float(ymax)))

    left=max(0,min(xmin,xmax))
    right=min(image_width,max(xmin,xmax))
    top=max(0,min(ymin,ymax))
    bottom=min(image_height,max(ymin,ymax))

    if right-left<1 or bottom-top<1:
        return None

    ratio=max(0.05,min(float(CLASSIFY_CENTER_RATIO),1.0))
    box_width=right-left
    box_height=bottom-top

    center_x=(left+right)/2
    center_y=(top+bottom)/2
    crop_width=max(1,box_width*ratio)
    crop_height=max(1,box_height*ratio)

    crop_left=int(round(center_x-crop_width/2))
    crop_right=int(round(center_x+crop_width/2))
    crop_top=int(round(center_y-crop_height/2))
    crop_bottom=int(round(center_y+crop_height/2))

    crop_left=max(left,min(crop_left,right-1))
    crop_right=max(crop_left+1,min(crop_right,right))
    crop_top=max(top,min(crop_top,bottom-1))
    crop_bottom=max(crop_top+1,min(crop_bottom,bottom))

    return (crop_left,crop_top,crop_right,crop_bottom)


def _get_rgb_mean(image):
    '''计算rgb均值'''
    stat=ImageStat.Stat(image)
    red,green,blue=stat.mean[:3]

    return (
        int(round(red)),
        int(round(green)),
        int(round(blue)),
    )


def _match_class_name(rgb_mean,rules=None):
    '''根据rgb均值匹配类别名'''
    if rules is None:
        rules=CLASSIFY_PIXEL_RULES

    for rule in rules:
        if _is_rgb_in_range(rgb_mean,rule):
            return rule["name"]

    return CLASSIFY_DEFAULT_CLASS_NAME


def _is_rgb_in_range(rgb_mean,rule):
    rgb_min=rule.get("rgb_min")
    rgb_max=rule.get("rgb_max")

    if rgb_min is None or rgb_max is None:
        return False

    for value,min_value,max_value in zip(rgb_mean,rgb_min,rgb_max):
        if value<min_value or value>max_value:
            return False

    return True
