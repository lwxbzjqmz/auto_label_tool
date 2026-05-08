import sys 
from pathlib import Path

BACKEND_DIR=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(BACKEND_DIR))

import os 
import json
import json
import time
from http.server import BaseHTTPRequestHandler,HTTPServer
from urllib.parse import urlparse

from config import (
    IMAGE_DIR,

    GROUNDING_DINO_WORKER_HOST,
    GROUNDING_DINO_WORKER_PORT,

    GROUNDING_DINO_MODEL_ID,
    GROUNDING_DINO_HF_HOME,
    GROUNDING_DINO_LOCAL_FILES_ONLY,

    GROUNDING_DINO_DEFAULT_PROMPT,
    GROUNDING_DINO_BOX_THRESHOLD,
    GROUNDING_DINO_TEXT_THRESHOLD,
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
    '''加载grounding NINO模型到GPU'''
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

def detect_image(image_name,prompt=None,box_threshold=None,text_threshold=None):
    '''检测单张图片,返回objects和boxes'''
    load_model()

    if prompt is None:
        prompt=GROUNDING_DINO_DEFAULT_PROMPT

    if box_threshold is None:
        box_threshold=GROUNDING_DINO_BOX_THRESHOLD

    if text_threshold is None:
        text_threshold=GROUNDING_DINO_TEXT_THRESHOLD

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
                "message":"Grounding NINO worker is running",
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
            )

            self._send_json(200,result)

        except Exception as e:
            self._send_json(500,{
                "success":False,
                "message":str(e),
            })

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
