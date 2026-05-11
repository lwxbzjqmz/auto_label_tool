import sys
from pathlib import Path

BACKEND_DIR=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(BACKEND_DIR))

import os
import json
import time
from http.server import BaseHTTPRequestHandler,HTTPServer
from urllib.parse import urlparse

from config import(
    IMAGE_DIR,

    GROUNDING_DINO_WORKER_HOST,
    GROUNDING_DINO_WORKER_PORT,

    GROUNDING_DINO_MODEL_ID,
    GROUNDING_DINO_HF_HOME,
    GROUNDING_DINO_LOCAL_FILES_ONLY,

    GROUNDING_DINO_DEFAULT_PROMPT,
    GROUNDING_DINO_BOX_THRESHOLD,
    GROUNDING_DINO_TEXT_THRESHOLD,

    GROUNDING_DINO_USE_NMS,
    GROUNDING_DINO_NMS_IOU_THRESHOLD,
)

HF_HOME_DIR=Path(GROUNDING_DINO_HF_HOME)
HF_HUB_DIR=HF_HOME_DIR/"hub"

os.environ["HF_HOME"]=str(HF_HOME_DIR)
os.environ["HF_HUB_CACHE"]=str(HF_HUB_DIR)
os.environ["TRANSFORMERS_CACHE"]=str(HF_HUB_DIR)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"]="1"

import torch
from PIL import Image
from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection


_processor=None
_model=None
_device=None


def load_model():
    '''加载Grounding DINO模型到GPU'''
    global _processor,_model,_device

    if _model is not None:
        return
    
    _device="cuda" if torch.cuda.is_available() else "cpu"

    print("="*60)
    print("Grounding DINO worker启动")
    print("model:",GROUNDING_DINO_MODEL_ID)
    print("device:",_device)
    print("HF_HOME:",HF_HOME_DIR)
    print("="*60)

    start=time.perf_counter()

    _processor=AutoProcessor.from_pretrained(
        GROUNDING_DINO_MODEL_ID,
        cache_dir=str(HF_HUB_DIR),
        local_files_only=GROUNDING_DINO_LOCAL_FILES_ONLY,
    )

    _model=AutoModelForZeroShotObjectDetection.from_pretrained(
        GROUNDING_DINO_MODEL_ID,
        cache_dir=str(HF_HUB_DIR),
        local_files_only=GROUNDING_DINO_LOCAL_FILES_ONLY,
    ).to(_device)

    _model.eval()

    if _device=="cuda":
        torch.cuda.synchronize()

    end=time.perf_counter()

    print(f"模型加载完成,耗时:{end-start:.3f}秒")
    print("="*60)


def detect_image(image_name,prompt=None,box_threshold=None,text_threshold=None,use_nms=None,nms_iou_threshold=None):
    '''检测单张图片,返回objects和boxes'''
    load_model()

    if prompt is None or prompt=="":
        prompt=GROUNDING_DINO_DEFAULT_PROMPT

    if box_threshold is None:
        box_threshold=GROUNDING_DINO_BOX_THRESHOLD
    else:
        box_threshold=float(box_threshold)

    if text_threshold is None:
        text_threshold=GROUNDING_DINO_TEXT_THRESHOLD
    else:
        text_threshold=float(text_threshold)

    if use_nms is None:
        use_nms=GROUNDING_DINO_USE_NMS
    else:
        use_nms=_to_bool(use_nms)

    if nms_iou_threshold is None:
        nms_iou_threshold=GROUNDING_DINO_NMS_IOU_THRESHOLD
    else:
        nms_iou_threshold=float(nms_iou_threshold)

    image_path=IMAGE_DIR/image_name

    if not image_path.exists():
        raise FileNotFoundError(f"image not found:{image_path}")
    
    image=Image.open(image_path).convert("RGB")

    infer_start=time.perf_counter()

    inputs=_processor(
        images=image,
        text=prompt,
        return_tensors="pt"
    ).to(_device)

    with torch.no_grad():
        outputs=_model(**inputs)

    if _device=="cuda":
        torch.cuda.synchronize()

    infer_end=time.perf_counter()

    post_start=time.perf_counter()

    results=_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]]
    )

    result=results[0]

    labels=result["text_labels"] if "text_labels" in result else result["labels"]

    objects=[]

    for box,score,label in zip(result["boxes"],result["scores"],labels):
        box=[int(round(x)) for x in box.tolist()]
        score=float(score.item())

        objects.append({
            "name":str(label),
            "box":box,
            "score":score,
        })

    if use_nms:
        objects=_nms_objects(objects,nms_iou_threshold)

    boxes=[obj["box"] for obj in objects]

    post_end=time.perf_counter()

    return{
        "success":True,
        "image_name":image_name,
        "objects":objects,
        "boxes":boxes,
        "box_count":len(boxes),
        "time":{
            "infer":round(infer_end-infer_start,3),
            "post":round(post_end-post_start,3),
            "total":round(post_end-infer_start,3),
        },
        "params":{
            "prompt":prompt,
            "box_threshold":box_threshold,
            "text_threshold":text_threshold,
            "use_nms":use_nms,
            "nms_iou_threshold":nms_iou_threshold,
        }
    }


class GroundingDinoHandler(BaseHTTPRequestHandler):
    def _send_json(self,status_code,data):
        body=json.dumps(data,ensure_ascii=False).encode("utf-8")

        self.send_response(status_code)
        self.send_header("Content-Type","application/json; charset=utf-8")
        self.send_header("Content-Length",str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path=urlparse(self.path).path

        if path=="/health":
            self._send_json(200,{
                "success":True,
                "message":"Grounding DINO worker is running",
                "model":GROUNDING_DINO_MODEL_ID,
                "device":_device,
            })
            return
        
        self._send_json(404,{
            "success":False,
            "message":"not found",
        })

    def do_POST(self):
        path=urlparse(self.path).path

        if path!="/detect":
            self._send_json(404,{
                "success":False,
                "message":"not found",
            })
            return

        try:
            content_length=int(self.headers.get("Content-Length",0))
            body=self.rfile.read(content_length).decode("utf-8")
            data=json.loads(body) if body else {}

            image_name=data.get("image_name")
            prompt=data.get("prompt")
            box_threshold=data.get("box_threshold")
            text_threshold=data.get("text_threshold")
            use_nms=data.get("use_nms")
            nms_iou_threshold=data.get("nms_iou_threshold")

            if not image_name:
                self._send_json(400,{
                    "success":False,
                    "message":"missing image_name",
                })
                return

            result=detect_image(
                image_name=image_name,
                prompt=prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                use_nms=use_nms,
                nms_iou_threshold=nms_iou_threshold,
            )

            self._send_json(200,result)

        except Exception as e:
            self._send_json(500,{
                "success":False,
                "message":str(e),
            })


def _nms_objects(objects,iou_threshold):
    '''对Grounding DINO输出做NMS'''
    if not objects:
        return []
    
    objects=sorted(
        objects,
        key=lambda item:item["score"],
        reverse=True
    )

    keep=[]

    while objects:
        best=objects.pop(0)
        keep.append(best)

        rest=[]

        for obj in objects:
            if _iou(best["box"],obj["box"])<iou_threshold:
                rest.append(obj)

        objects=rest

    return keep


def _iou(box_a,box_b):
    '''计算两个box的IoU'''
    ax1,ay1,ax2,ay2=box_a
    bx1,by1,bx2,by2=box_b

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


def _to_bool(value):
    '''把前端传来的值转成bool,用于判定是否使用NMS'''
    if isinstance(value,bool):
        return value
    
    if isinstance(value,str):
        value=value.strip().lower()

        if value in ("true","1","yes","on"):
            return True
        
        if value in ("false","0","no","off"):
            return False
    
    return bool(value)


def main():
    load_model()

    server=HTTPServer(
        (GROUNDING_DINO_WORKER_HOST,GROUNDING_DINO_WORKER_PORT),
        GroundingDinoHandler
    )

    print(f"worker监听地址:http://{GROUNDING_DINO_WORKER_HOST}:{GROUNDING_DINO_WORKER_PORT}")
    print("按 Ctrl+C 停止 worker")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("worker停止")
    finally:
        server.server_close()


if __name__=="__main__":
    main()