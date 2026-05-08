import sys
from pathlib import Path

BACKEND_DIR=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(BACKEND_DIR))

from config import(
    IMAGE_DIR,
    LARGE_MODEL_DEFAULT_RESPONSE_PATH,
)

from xml_writer import write_voc_xml
from large_model.response_parser import load_large_model_json,parse_large_model_response

def run_large_model_json_to_xml(json_path=None):
    '''读取JOSN,返回VOC XML'''
    if json_path is None:
        json_path=LARGE_MODEL_DEFAULT_RESPONSE_PATH

    json_path=Path(json_path)

    data=load_large_model_json(json_path)
    image_boxes_map=parse_large_model_response(data)

    total_images=0
    saved_images=0
    skipped_images=0
    total_boxes=0

    for image_name,boxes in image_boxes_map.items():
        total_images+=1

        image_path=IMAGE_DIR/image_name
        
        if not image_path.exists():
            print(f"[跳过]图片不存在:{image_name}")
            skipped_images+=1
            continue

        result=write_voc_xml(image_name,boxes)

        saved_images+=1
        total_boxes+=result["box_count"]

        print(f"[保存]{result['xml_name']} 框数量:{result['box_count']}")

    print("="*60)
    print("large_model JSON 导入完成")
    print(f"JSON文件:{json_path}")
    print(f"总图片数:{total_images}")
    print(f"成功保存:{saved_images}")
    print(f"跳过图片:{skipped_images}")
    print(f"总框数量:{total_boxes}")
    print("="*60)

if __name__=="__main__":
    if len(sys.argv)>=2:
        run_large_model_json_to_xml(sys.argv[1])
    else:
        run_large_model_json_to_xml()
        