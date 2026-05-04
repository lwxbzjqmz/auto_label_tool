from PIL import Image
import numpy as np

from config import(
    YOLOV3_MBV2_CROP_X,
    YOLOV3_MBV2_CROP_Y,
    YOLOV3_MBV2_CROP_W,
    YOLOV3_MBV2_CROP_H,

    YOLOV3_MBV2_MODEL_INPUT_W,
    YOLOV3_MBV2_MODEL_INPUT_H,
)

CROP_X=YOLOV3_MBV2_CROP_X
CROP_Y=YOLOV3_MBV2_CROP_Y
CROP_W=YOLOV3_MBV2_CROP_W
CROP_H=YOLOV3_MBV2_CROP_H

MODEL_INPUT_W=YOLOV3_MBV2_MODEL_INPUT_W
MODEL_INPUT_H=YOLOV3_MBV2_MODEL_INPUT_H

def load_and_crop_image(image_path):
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

def prepare_input_tensor(crop_img,input_detail):
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