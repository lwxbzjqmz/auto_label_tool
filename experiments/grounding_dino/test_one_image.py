from pathlib import Path
import json
import os
import time

HF_CACHE_DIR=Path(r"C:\Users\lwx\.cache\huggingface")

os.environ["HF_HOME"]=str(HF_CACHE_DIR)
os.environ["HF_HUB_CACHE"]=str(HF_CACHE_DIR/"hub")
os.environ["TRANSFORMERS_CACHE"]=str(HF_CACHE_DIR/"hub")

LOCAL_FILES_ONLY=True

import torch
from PIL import Image
from transformers import AutoProcessor,AutoModelForZeroShotObjectDetection

PROJECT_DIR=Path(__file__).resolve().parent.parent.parent

IMAGE_PATH=PROJECT_DIR/"data"/"JPEGImages"/"03598.jpg"

MODEL_ID="IDEA-Research/grounding-dino-base"

TEXT_PROMPT="tennis ball. blue bean bag. red bean bag. white teddy bear. brown teddy bear"

BOX_THRESHOLD=0.25
TEXT_THRESHOLD=0.20

def main():
    total_start=time.perf_counter()

    print("image:",IMAGE_PATH)
    print("image exists:",IMAGE_PATH.exists())

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"image not found:{IMAGE_PATH}")

    device="cuda" if torch.cuda.is_available() else "cpu"
    print("device:",device)

    load_start=time.perf_counter()

    processor=AutoProcessor.from_pretrained(
        MODEL_ID,
        cache_dir=str(HF_CACHE_DIR/"hub"),
        local_files_only=LOCAL_FILES_ONLY
    )

    model=AutoModelForZeroShotObjectDetection.from_pretrained(
        MODEL_ID,
        cache_dir=str(HF_CACHE_DIR/"hub"),
        local_files_only=LOCAL_FILES_ONLY
    ).to(device)

    model.eval()#切换到推理模式

    if device=="cuda":
        torch.cuda.synchronize()

    load_end=time.perf_counter()

    image_start=time.perf_counter()

    image=Image.open(IMAGE_PATH).convert("RGB")

    image_end=time.perf_counter()

    infer_start=time.perf_counter()

    inputs=processor(
        images=image,
        text=TEXT_PROMPT,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():#关闭梯度计算
        outputs=model(**inputs)

    if device=="cuda":
        torch.cuda.synchronize()

    infer_end=time.perf_counter()

    post_start=time.perf_counter()

    results=processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        target_sizes=[image.size[::-1]]
    )

    result=results[0]

    objects=[]

    for box,score,label in zip(result["boxes"],result["scores"],result["labels"]):
        box=[int(round(x)) for x in box.tolist()]
        score=float(score.item())

        objects.append({
            "name":str(label),
            "box":box,
            "score":score
        })

    output={
        "image_name":IMAGE_PATH.name,
        "objects":objects
    }

    post_end=time.perf_counter()

    print(json.dumps(output,ensure_ascii=False,indent=4))

    total_end=time.perf_counter()

    print("="*60)
    print(f"模型加载耗时:{load_end-load_start:.3f} 秒")
    print(f"图片读取耗时:{image_end-image_start:.3f} 秒")
    print(f"模型推理耗时:{infer_end-infer_start:.3f} 秒")
    print(f"后处理耗时:{post_end-post_start:.3f} 秒")
    print(f"总耗时:{total_end-total_start:.3f} 秒")
    print("="*60)

if __name__=="__main__":
    main()