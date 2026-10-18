#此文件用于自定义批量标注(Grounding DINO 检测并保存),本人认为该操作不应该放在前端
#运行: python batch_detector_and_save.py

from detector import get_detect_result
from xml_writer import write_voc_xml
from config import IMAGE_DIR
from pathlib import Path

first_image_name="05921"
last_image_name="06200"

# Grounding DINO 检测框阈值(box_threshold),可按需修改。
GROUNDING_DINO_BOX_THRESHOLD=0.5

def save_voc_xml(image_name):
    image_path=IMAGE_DIR/Path(image_name).with_suffix(".jpg").name

    detect_result=get_detect_result(
        image_path,
        detector="grounding_dino",
        grounding_dino_options={
            "box_threshold":GROUNDING_DINO_BOX_THRESHOLD,
        },
    )

    class_boxes=detect_result.get("class_boxes",[])

    write_voc_xml(image_name,class_boxes)

def batch_save(first_image_name,last_image_name):
    image_names=[]

    for i in range(int(first_image_name),int(last_image_name)+1):
        image_names.append(f"{i:05}.jpg")

    for image_name in image_names:
        try: 
            save_voc_xml(image_name)
            print(f"标注{image_name}")
        except (RuntimeError,FileNotFoundError):
            continue


if __name__=="__main__":
    print(f"开始批标注(阈值为 {GROUNDING_DINO_BOX_THRESHOLD})")
    batch_save(first_image_name,last_image_name)
    print(f"批标注结束")
