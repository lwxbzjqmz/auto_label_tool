from grounding_dino.client import detect_by_grounding_dino_worker

def detect_by_large_model(image_path,options=None):
    '''使用Grounding DINO worker进行自动标注,返回完整结果'''
    result=detect_by_grounding_dino_worker(
        image_path,
        options=options
    )

    return result