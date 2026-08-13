#此文件用于自定义批量标注，本人认为改操作不应该放在前端

from detector import detect_by_large_model
from xml_writer import write_voc_xml
#print(detect_by_large_model("03598.jpg"))
first_image_name="03191"
last_image_name="04000"

def save_voc_xml(image_name):
    data=detect_by_large_model(image_name)

    image_name=data.get("image_name")
    boxes=data.get("boxes",[])

    write_voc_xml(image_name,boxes)

def batch_save(first_image_name,last_image_name):
    image_names=[]

    for i in range(int(first_image_name),int(last_image_name)+1):
        image_names.append(f"{i:05}.jpg")

    for image_name in image_names:
        try: 
            save_voc_xml(image_name)
            print(f"标注{image_name}")
        except RuntimeError:
            continue


if __name__=="__main__":
    print(f"开始批标注")
    batch_save(first_image_name,last_image_name)
    print(f"批标注结束")