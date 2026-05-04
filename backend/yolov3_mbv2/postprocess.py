import math
import numpy as np

from config import(
    YOLOV3_MBV2_CROP_X,
    YOLOV3_MBV2_CROP_Y,
    YOLOV3_MBV2_CROP_W,
    YOLOV3_MBV2_CROP_H,

    YOLOV3_MBV2_MODEL_INPUT_W,
    YOLOV3_MBV2_MODEL_INPUT_H,

    YOLOV3_MBV2_ORIGINAL_IMAGE_W,
    YOLOV3_MBV2_ORIGINAL_IMAGE_H,
)

CROP_X=YOLOV3_MBV2_CROP_X
CROP_Y=YOLOV3_MBV2_CROP_Y
CROP_W=YOLOV3_MBV2_CROP_W
CROP_H=YOLOV3_MBV2_CROP_H

MODEL_INPUT_W=YOLOV3_MBV2_MODEL_INPUT_W
MODEL_INPUT_H=YOLOV3_MBV2_MODEL_INPUT_H

ORIGINAL_IMAGE_W=YOLOV3_MBV2_ORIGINAL_IMAGE_W
ORIGINAL_IMAGE_H=YOLOV3_MBV2_ORIGINAL_IMAGE_H

def decode_yolo_output(raw_output,output_detail,anchors,score_threshold):
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

def nms(boxes_with_scores,iou_threshold):
    '''非极大值抑制,去掉高度重叠的框'''
    if not boxes_with_scores:
        return []
    
    boxes=sorted(
        boxes_with_scores,
        key=lambda item:item[4],
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

def map_boxes_to_original_image(boxes_with_scores):
    '''把模型输入坐标映射回原始图片坐标'''
    boxes=[]

    scale_x=CROP_W/MODEL_INPUT_W
    scale_y=CROP_H/MODEL_INPUT_H

    for item in boxes_with_scores:
        xmin,ymin,xmax,ymax,score=item

        xmin_crop=int(round(xmin*scale_x))
        ymin_crop=int(round(ymin*scale_y))
        xmax_crop=int(round(xmax*scale_x))
        ymax_crop=int(round(ymax*scale_y))

        xmin_full=xmin_crop+CROP_X
        ymin_full=ymin_crop+CROP_Y
        xmax_full=xmax_crop+CROP_X
        ymax_full=ymax_crop+CROP_Y

        xmin_full=_clamp(xmin_full,0,ORIGINAL_IMAGE_W)
        xmax_full=_clamp(xmax_full,0,ORIGINAL_IMAGE_W)
        ymin_full=_clamp(ymin_full,0,ORIGINAL_IMAGE_H)
        ymax_full=_clamp(ymax_full,0,ORIGINAL_IMAGE_H)

        if xmax_full-xmin_full>=1 and ymax_full-ymin_full>=1:
            boxes.append([xmin_full,ymin_full,xmax_full,ymax_full])

    return boxes

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