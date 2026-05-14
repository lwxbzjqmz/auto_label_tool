import json
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.error import HTTPError,URLError

from config import(
    GROUNDING_DINO_WORKER_URL,
    GROUNDING_DINO_DEFAULT_PROMPT,
    GROUNDING_DINO_BOX_THRESHOLD,
    GROUNDING_DINO_TEXT_THRESHOLD,
    GROUNDING_DINO_USE_NMS,
    GROUNDING_DINO_NMS_IOU_THRESHOLD,
    GROUNDING_DINO_REQUEST_TIMEOUT,
)


def detect_by_grounding_dino_worker(image_path,options=None):
    '''请求Grounding DINO worker,返回完整检测结果'''
    image_path=Path(image_path)

    payload={
        "image_name":image_path.name,
        "prompt":GROUNDING_DINO_DEFAULT_PROMPT,
        "box_threshold":GROUNDING_DINO_BOX_THRESHOLD,
        "text_threshold":GROUNDING_DINO_TEXT_THRESHOLD,
        "use_nms":GROUNDING_DINO_USE_NMS,
        "nms_iou_threshold":GROUNDING_DINO_NMS_IOU_THRESHOLD,
    }

    if options:
        for key,value in options.items():
            if value is not None:
                payload[key]=value

    data=json.dumps(payload,ensure_ascii=False).encode("utf-8")

    request=Request(
        GROUNDING_DINO_WORKER_URL,
        data=data,
        headers={
            "Content-Type":"application/json"
        },
        method="POST",
    )

    try:
        with urlopen(request,timeout=GROUNDING_DINO_REQUEST_TIMEOUT) as response:
            response_text=response.read().decode("utf-8")
            result=json.loads(response_text)

    except HTTPError as e:
        error_text=e.read().decode("utf-8",errors="ignore")
        raise RuntimeError(f"Grounding DINO worker HTTP错误:{e.code}, {error_text}")

    except URLError as e:
        raise RuntimeError(
            "无法连接Grounding DINO worker,请先启动 worker.py。"
            f" 原因:{e}"
        )

    if not result.get("success"):
        raise RuntimeError(f"Grounding DINO worker检测失败:{result.get('message')}")

    return result