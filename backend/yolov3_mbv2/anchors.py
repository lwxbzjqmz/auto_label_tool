from config import YOLOV3_MBV2_ANCHORS_PATH

_anchors=None

def load_anchors():
    '''读取anchors文件,返回元组列表'''
    global _anchors

    if _anchors is not None:
        return _anchors
    
    if not YOLOV3_MBV2_ANCHORS_PATH.exists():
        raise FileNotFoundError(
            f"YOLOv3-mbv2 anchors not found: {YOLOV3_MBV2_ANCHORS_PATH}"
        )
    
    text=YOLOV3_MBV2_ANCHORS_PATH.read_text(encoding="utf-8").strip()
    values=[]

    for part in text.replace("\n",",").split(","):
        part=part.strip()
        if part:
            values.append(float(part))

    if len(values)%2!=0:
        raise ValueError(f"anchors 数量不成对: {text}")

    anchors=[]

    for i in range(0,len(values),2):
        anchors.append((values[i],values[i+1]))

    if len(anchors)!=3:
        raise ValueError(f"anchors数据非3组 {len(anchors)} 组: {anchors}")

    _anchors=anchors
    return _anchors