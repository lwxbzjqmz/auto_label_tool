from pathlib import Path
import shutil
import time

from config import(
    IMAGE_DIR,
    ANNOTATION_DIR,
    REJECTED_IMAGE_DIR,
    REJECTED_ANNOTATION_DIR,
)

def reject_image(image_name):
    '''移除图片和VOC XML'''
    image_path=(IMAGE_DIR/image_name).resolve()

    try:
        image_path.relative_to(IMAGE_DIR.resolve())
    except ValueError:
        raise ValueError("Invalid image path")
    
    if not image_path.exists() or not image_path.is_file():
        raise FileNotFoundError(f"image not found:{image_name}")

    REJECTED_IMAGE_DIR.mkdir(parents=True,exist_ok=True)
    REJECTED_ANNOTATION_DIR.mkdir(parents=True,exist_ok=True)

    rejected_image_path=_make_unique_path(REJECTED_IMAGE_DIR/image_path.name)

    shutil.move(str(image_path),str(rejected_image_path))

    xml_name=Path(image_name).with_suffix(".xml").name
    xml_path=ANNOTATION_DIR/xml_name

    moved_xml=False
    rejected_xml_path=None

    if xml_path.exists() and xml_path.is_file():
        rejected_xml_path=_make_unique_path(REJECTED_ANNOTATION_DIR/xml_name)
        shutil.move(str(xml_path),str(rejected_xml_path))
        moved_xml=True

    return {
        "image_name":image_name,
        "moved_image":True,
        "moved_xml":moved_xml,
        "rejected_image_path":str(rejected_image_path),
        "rejected_xml_path":str(rejected_xml_path) if rejected_xml_path else None,
    } 

def _make_unique_path(path):
    '''避免覆盖废弃区已有文件'''
    path=Path(path)

    if not path.exists():
        return path
    
    timestamp=time.strftime("%Y%m%d_%H%M%S")
    
    return path.with_name(f"{path.stem}_{timestamp}{path.suffix}")


