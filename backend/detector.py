from pathlib import Path
from PIL import Image

def get_boxes(image_path):
    '''临时自动标注函数'''
    image_path=Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found:{image_path}")
    
    with Image.open(image_path) as img:
        width,height=img.size

    #先进行测试
    box1=[
        int(width*0.10),
        int(height*0.15),
        int(width*0.35),
        int(height*0.45),
    ]

    box2=[
        int(width*0.20),
        int(height*0.15),
        int(width*0.35),
        int(height*0.45),
    ]

    return box1,box2