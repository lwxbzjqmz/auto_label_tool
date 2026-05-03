from pathlib import Path
import xml.etree.ElementTree as ET
from config import ANNOTATION_DIR

def read_voc_xml(image_name):
    '''读voc xml,返回boxes'''
    xml_name=Path(image_name).with_suffix(".xml").name
    xml_path=ANNOTATION_DIR/xml_name

    if not xml_path.exists():
        return{
            "exists":False,
            "xml_name":xml_name,
            "boxes":[]
        }
    
    try:
        tree=ET.parse(xml_path)
        root=tree.getroot()
    except ET.ParseError as e:
        raise ValueError(f"XML parse error: {xml_path},{e}")
    
    boxes=[]

    for obj in root.findall("object"):
        bndbox=obj.find("bndbox")

        if bndbox is None:
            continue

        xmin_el=bndbox.find("xmin")
        ymin_el=bndbox.find("ymin")
        xmax_el = bndbox.find("xmax")
        ymax_el = bndbox.find("ymax")

        if None in [xmin_el,ymin_el,xmax_el,ymax_el]:
            continue

        try:
            xmin=int(float(xmin_el.text))
            ymin=int(float(ymin_el.text))
            xmax = int(float(xmax_el.text))
            ymax = int(float(ymax_el.text))
        except(TypeError,ValueError):
            continue

        boxes.append([xmin,ymin,xmax,ymax])

    return{
        "exists":True,
        "xml_name":xml_name,
        "boxes":boxes
    }

