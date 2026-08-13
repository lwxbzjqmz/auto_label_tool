from pathlib import Path
import xml.etree.ElementTree as ET
from config import ANNOTATION_DIR


def read_voc_xml(image_name):
    '''读voc xml,返回class_boxes'''
    xml_name = Path(image_name).with_suffix(".xml").name
    xml_path = ANNOTATION_DIR / xml_name

    if not xml_path.exists():
        return {
            "exists": False,
            "xml_name": xml_name,
            "class_boxes": []
        }

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        raise ValueError(f"XML parse error: {xml_path},{e}")

    class_boxes = []

    for obj in root.findall("object"):
        class_name_el = obj.find("name")
        bndbox = obj.find("bndbox")

        if class_name_el is None or bndbox is None:
            continue

        xmin_el = bndbox.find("xmin")
        ymin_el = bndbox.find("ymin")
        xmax_el = bndbox.find("xmax")
        ymax_el = bndbox.find("ymax")

        if None in [xmin_el, ymin_el, xmax_el, ymax_el]:
            continue

        try:
            class_name = (class_name_el.text or "object").strip() or "object"
            xmin = int(float(xmin_el.text))
            ymin = int(float(ymin_el.text))
            xmax = int(float(xmax_el.text))
            ymax = int(float(ymax_el.text))
        except (TypeError, ValueError):
            continue

        class_boxes.append([class_name, xmin, ymin, xmax, ymax])

    return {
        "exists": True,
        "xml_name": xml_name,
        "class_boxes": class_boxes
    }
