from pathlib import Path
import xml.etree.ElementTree as ET
from xml.dom import minidom
from PIL import Image
from config import IMAGE_DIR,ANNOTATION_DIR,CLASS_NAME,VOC_FOLDER_NAME

def _prettify_xml(element):
    '''调整缩进,prettify文件'''
    rough_string=ET.tostring(element,encoding="utf-8")
    parsed=minidom.parseString(rough_string)
    pretty=parsed.toprettyxml(indent="    ",encoding="utf-8")

    #去掉声明
    text=pretty.decode("utf-8")
    lines=text.splitlines()
    if lines and lines[0].startswith("<?xml"):
        lines=lines[1:]
    
    #去掉空行
    lines=[line for line in lines if line.strip()]
    return "\n".join(lines)+"\n"
def _normalize_box(box,image_width,image_height):
    '''清洗坐标'''
    if len(box)!=4:
        return None
    xmin,ymin,xmax,ymax=box

    xmin=int(round(round(float(xmin))))
    ymin=int(round(float(ymin)))
    xmax=int(round(float(xmax)))
    ymax=int(round(float(ymax)))

    xmin=max(0,min(xmin,image_width))
    xmax=max(0,min(xmax,image_width))
    ymin = max(0, min(ymin, image_height))
    ymax = max(0, min(ymax, image_height))

    left = min(xmin, xmax)
    right = max(xmin, xmax)
    top = min(ymin, ymax)
    bottom = max(ymin, ymax)

    if right-left<1 or bottom-top<1:
        return None
    
    return [left,top,right,bottom]

def write_voc_xml(image_name,boxes):
    '''生成 voc xml'''
    image_path=IMAGE_DIR/image_name

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found:{image_path}")
    
    ANNOTATION_DIR.mkdir(parents=True,exist_ok=True)

    with Image.open(image_path) as img:
        image_width,image_height=img.size
        image_depth=len(img.getbands())

    annotation=ET.Element("annotation")

    folder=ET.SubElement(annotation,"folder")
    folder.text=VOC_FOLDER_NAME

    filename=ET.SubElement(annotation,"path")
    filename.text=image_name

    path=ET.SubElement(annotation,"path")
    path.text=str(image_path)

    source=ET.SubElement(annotation,"source")
    database=ET.SubElement(annotation,"database")
    database.text="Unknown"

    size=ET.SubElement(annotation,"size")

    width = ET.SubElement(size, "width")
    width.text = str(image_width)

    height = ET.SubElement(size, "height")
    height.text = str(image_height)

    depth = ET.SubElement(size, "depth")
    depth.text = str(image_depth)

    segmented = ET.SubElement(annotation, "segmented")
    segmented.text = "0"

    valid_boxes=[]

    for box in boxes:
        normalized_box=_normalize_box(box,image_width,image_height)

        if normalized_box is None:
            continue

        xmin, ymin, xmax, ymax = normalized_box
        valid_boxes.append(normalized_box)

        obj = ET.SubElement(annotation, "object")

        name = ET.SubElement(obj, "name")
        name.text = CLASS_NAME

        pose = ET.SubElement(obj, "pose")
        pose.text = "Unspecified"

        truncated = ET.SubElement(obj, "truncated")
        truncated.text = "0"

        difficult = ET.SubElement(obj, "difficult")
        difficult.text = "0"

        bndbox = ET.SubElement(obj, "bndbox")

        xmin_el = ET.SubElement(bndbox, "xmin")
        xmin_el.text = str(xmin)

        ymin_el = ET.SubElement(bndbox, "ymin")
        ymin_el.text = str(ymin)

        xmax_el = ET.SubElement(bndbox, "xmax")
        xmax_el.text = str(xmax)

        ymax_el = ET.SubElement(bndbox, "ymax")
        ymax_el.text = str(ymax)

    xml_text = _prettify_xml(annotation)

    xml_name = Path(image_name).with_suffix(".xml").name
    xml_path = ANNOTATION_DIR / xml_name

    xml_path.write_text(xml_text, encoding="utf-8")

    return {
        "xml_name": xml_name,
        "xml_path": str(xml_path),
        "box_count": len(valid_boxes),
        "image_width": image_width,
        "image_height": image_height,
        "image_depth": image_depth
    }






    