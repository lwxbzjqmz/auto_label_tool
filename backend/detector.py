from pathlib import Path
from PIL import Image
from config import MODEL_SELECT,YOLOV3_MBV2_PC_MODEL_PATH
import math
import numpy as np
import tensorflow as tf
from config import(
    MODEL_SELECT,
    YOLOV3_MBV2_PC_MODEL_PATH,
    YOLOV3_MBV2_ANCHORS_PATH,
)

CROP_X=40
CROP_Y=0
CROP_W=240
CROP_H=240

MODEL_INPUT_W=144
MODEL_INPUT_H=144

SCORE_THRESHOLD=0.6 #置信度阈值
NMS_IOU_THRESHOLD=0.3#NMS阈值

_interpreter=None
_input_details=None
_output_details=None
_anchors=None

def get_boxes(image_path):
    '''自动模型选择'''
    if MODEL_SELECT==0:
        return get_boxes_by_large_model(image_path)
    else:
        return get_boxes_by_yolov3_mbv2(image_path)
    
def get_boxes_by_large_model(image_path):
    '''使用大模型'''
    raise NotImplementedError("large_model 等待开发")

def get_boxes_by_yolov3_mbv2(image_path):
    '''调用自己的yolov3_mbv2模型'''
    boxes=[]

    image_path=Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    if not YOLOV3_MBV2_PC_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"YOLOv3-mbv2 model not found: {YOLOV3_MBV2_PC_MODEL_PATH}"
        )
    
    if not YOLOV3_MBV2_ANCHORS_PATH.exists():
        raise FileNotFoundError(
            f"YOLOv3-mbv2 anchors not found: {YOLOV3_MBV2_ANCHORS_PATH}"
        )   

    interpreter,input_details,output_details=_get_interpreter()
    anchors=_get_anchors()

    crop_img=_load_and_crop_image(image_path)

    input_tensor=_prepare_input_tensor(crop_img,input_details[0])

    interpreter.set_tensor(input_details[0]["index"], input_tensor)
    interpreter.invoke()    

    raw_output = interpreter.get_tensor(output_details[0]["index"])

    boxes_with_scores = _decode_yolo_output(
        raw_output=raw_output,
        output_detail=output_details[0],
        anchors=anchors,
        score_threshold=SCORE_THRESHOLD,
    )

    boxes_with_scores = _nms(
        boxes_with_scores,
        iou_threshold=NMS_IOU_THRESHOLD,
    )

    boxes=[]

    for item in boxes_with_scores:
        xmin,ymin,xmax,ymax,score=item

        scale_x=CROP_W/MODEL_INPUT_W
        scale_y=CROP_H/MODEL_INPUT_H

        xmin_crop=int(round(xmin*scale_x))
        ymin_crop=int(round(ymin*scale_y))
        xmax_crop=int(round(xmax*scale_x))
        ymax_crop=int(round(ymax*scale_y))

        xmin_full=xmin_crop+CROP_X # 再映射回原始320x240图像坐标
        ymin_full=ymin_crop+CROP_Y
        xmax_full=xmax_crop+CROP_X
        ymax_full=ymax_crop+CROP_Y

        xmin_full=_clamp(xmin_full,0,320)#限制到原图的范围内
        xmax_full=_clamp(xmax_full,0,320)
        ymin_full=_clamp(ymin_full,0,240)
        ymax_full=_clamp(ymax_full,0,240)
    
        if xmax_full-xmin_full>=1 and ymax_full-ymin_full>=1:
            boxes.append([xmin_full, ymin_full, xmax_full, ymax_full])

    return boxes

def _get_interpreter():
    '''加载 tflite interpreter'''
    global _interpreter, _input_details, _output_details

    if _interpreter is not None:
        return _interpreter, _input_details, _output_details
    
    _interpreter=tf.lite.Interpreter(
        model_path=str(YOLOV3_MBV2_PC_MODEL_PATH)
    )

    _interpreter.allocate_tensors()

    _input_details=_interpreter.get_input_details()
    _output_details=_interpreter.get_output_details()

    return _interpreter,_input_details,_output_details

def _get_anchors():
    '''读取anchors文件,返回元组列表'''
    global _anchors

    if _anchors is not None:
        return _anchors
    
    text=YOLOV3_MBV2_ANCHORS_PATH.read_text(encoding="utf-8").strip()
    values=[]

    for part in text.replace("\n", ",").split(","):
        part=part.strip()
        if part:
            values.append(float(part))

    if len(values) % 2 !=0:
        raise ValueError(f"anchors 数量不成对: {text}")

    anchors=[]

    for i in range(0, len(values), 2):
        anchors.append((values[i], values[i + 1]))

    if len(anchors) != 3:
        raise ValueError(f"anchors数据非3组 {len(anchors)} 组: {anchors}")

    _anchors=anchors
    return _anchors

def _load_and_crop_image(image_path):
    '''读取jpg,并裁剪'''
    with Image.open(image_path) as img:
        img=img.convert("RGB")
        
        crop_img=img.crop((
            CROP_X,
            CROP_Y,
            CROP_X+CROP_W,
            CROP_Y+CROP_H,
        ))

    return crop_img

def _prepare_input_tensor(crop_img,input_detail):
    '''归一化,INT8量化,维度调整'''
    resized=crop_img.resize((MODEL_INPUT_W,MODEL_INPUT_H),Image.BILINEAR)

    arr=np.asarray(resized,dtype=np.uint8)

    scale,zero_point=input_detail["quantization"]

    if input_detail["dtype"]==np.int8:
        #real_value=pixel/255
        #quantized=real_value/scale+zero_point
        arr_float=arr.astype(np.float32)/255.0
        arr_q=np.round(arr_float/scale+zero_point)
        arr_q=np.clip(arr_q,-128,127).astype(np.int8)
        return np.expand_dims(arr_q,axis=0)
    
    if input_detail["dtype"]==np.uint8:
        arr_float=arr.astype(np.float32)/255.0
        arr_q=np.round(arr_float/scale+zero_point)
        arr_q=np.clip(arr_q,0,255).astype(np.uint8)
        return np.expand_dims(arr_q,axis=0)
    
    arr_float=arr.astype(np.float32)/255.0
    return np.expand_dims(arr_float,axis=0).astype(np.float32)#float32模型兜底

def _decode_yolo_output(raw_output,output_detail,anchors,score_threshold):
    '''解码yolo输出'''
    output=raw_output[0]
    if output.dtype==np.int8 or output.dtype==np.uint8:
        scale,zero_point=output_detail["quantization"]
        output=(output.astype(np.float32)-zero_point)*scale
    else:
        output=output.astype(np.float32)

    grid_h,grid_w,channels=output.shape

    num_anchors=len(anchors)
    values_per_anchor=channels//num_anchors

    if values_per_anchor<6:
        raise ValueError(
             f"输出通道无法按 YOLO 格式解码: channels={channels}, anchors={num_anchors}"
        )

    output=output.reshape((grid_h,grid_w,num_anchors,values_per_anchor))

    boxes=[]

    for gy in range(grid_h):
        for gx in range(grid_w):
            for anchor_index,anchor in enumerate(anchors):
                pred=output[gy,gx,anchor_index]

                tx=pred[0]
                ty=pred[1]
                tw=pred[2]
                th=pred[3]
                obj_logit=pred[4]
                cls_logit=pred[5]

                objectness=_sigmoid(obj_logit)
                class_score=_sigmoid(cls_logit)

                score=objectness*class_score

                if score<score_threshold:
                    continue

                anchor_w,anchor_h=anchor

                bx=(_sigmoid(tx)+gx)/grid_w*MODEL_INPUT_W
                by=(_sigmoid(ty)+gy)/grid_h*MODEL_INPUT_H

                # 防止 exp 溢出
                tw=float(np.clip(tw,-10,10))
                th=float(np.clip(th,-10,10))

                bw=anchor_w*math.exp(tw)
                bh=anchor_h*math.exp(th)

                xmin=bx-bw/2
                ymin=by-bh/2
                xmax=bx+bw/2
                ymax=by+bh/2

                xmin=_clamp(xmin,0,MODEL_INPUT_W)
                xmax=_clamp(xmax,0,MODEL_INPUT_W)
                ymin=_clamp(ymin,0,MODEL_INPUT_H)
                ymax=_clamp(ymax,0,MODEL_INPUT_H)

                if xmax-xmin>=1 and ymax-ymin>=1:
                    boxes.append([xmin,ymin,xmax,ymax,score])

    return boxes
    
def _nms(boxes_with_scores,iou_threshold):
    '''非极大性抑制，去掉高度重叠的框'''
    if not boxes_with_scores:
        return[]
    
    boxes=sorted(
        boxes_with_scores,
        key=lambda item: item[4],
        reverse=True
    )

    keep=[]

    while boxes:
        best=boxes.pop(0)
        keep.append(best)

        rest=[]

        for box in boxes:
            if _iou(best,box)<iou_threshold:
                rest.append(box)

        boxes=rest

    return keep

def _iou(box_a,box_b):
    '''计算两个框的IoU'''
    ax1,ay1,ax2,ay2=box_a[:4]
    bx1,by1,bx2,by2=box_b[:4]

    inter_x1=max(ax1,bx1)
    inter_y1=max(ay1,by1)
    inter_x2=min(ax2,bx2)
    inter_y2=min(ay2,by2)

    inter_w=max(0,inter_x2-inter_x1)
    inter_h=max(0,inter_y2-inter_y1)

    inter_area=inter_w*inter_h

    area_a=max(0,ax2-ax1)*max(0,ay2-ay1)
    area_b=max(0,bx2-bx1)*max(0,by2-by1)

    union=area_a+area_b-inter_area

    if union<=0:
        return 0
    
    return inter_area/union

def _sigmoid(x):
    '''输出归一化,激活函数'''
    x=float(np.clip(x,-50,50))
    return 1.0/(1.0+math.exp(-x))

def _clamp(value,min_value,max_value):
    return max(min_value,min(max_value,value))


