from grounding_dino.client import get_boxes_by_grounding_dino_worker


def get_boxes_by_large_model(image_path):
    '''使用Grounding DINO worker进行自动标注'''
    boxes=get_boxes_by_grounding_dino_worker(image_path)

    return boxes
