from config import (
    CLASSIFY_MODEL_ROOT,
    CLASSIFY_MODEL_FILENAME,
    CLASSIFY_MODEL_SELECT,
    MODEL_CLASSIFY_PATH,
)


def normalize_model_select(model_select=None):
    '''处理分类模型编号'''
    if model_select is None or model_select == '' or model_select == 'default':
        return CLASSIFY_MODEL_SELECT
    return model_select


def _find_tflite_path(model_dir):
    '''在模型目录中查找 tflite 文件,优先使用配置的文件名'''
    preferred = model_dir / CLASSIFY_MODEL_FILENAME
    if preferred.exists():
        return preferred

    tflite_files = sorted(model_dir.glob('*.tflite'))
    if tflite_files:
        return tflite_files[0]

    return None


def get_classify_model_path(model_select=None):
    '''获取指定分类模型路径

    支持两种模型目录结构:
    1. 根目录平铺: models/classify/1.tflite, 2.tflite, ...(与前端编号选择对应)
    2. 子目录(与 yolov3 一致): models/classify/1/xxx.tflite
    兼容旧命名 models/classify/classify.tflite
    '''
    model_select = normalize_model_select(model_select)

    # 1. 根目录平铺文件: models/classify/<select>.tflite
    root_file = CLASSIFY_MODEL_ROOT / f'{model_select}.tflite'
    if root_file.exists():
        return root_file

    # 2. 子目录: models/classify/<select>/xxx.tflite
    model_dir = CLASSIFY_MODEL_ROOT / str(model_select)
    path = _find_tflite_path(model_dir)
    if path is not None:
        return path

    # 3. 旧命名兼容: models/classify/classify.tflite
    if MODEL_CLASSIFY_PATH.exists():
        return MODEL_CLASSIFY_PATH

    return root_file


def list_classify_models():
    '''列出可用分类模型

    扫描 models/classify 根目录下的 *.tflite(如 1.tflite, 2.tflite, 3.tflite)
    以及各数字子目录(与 yolov3 一致)。根目录文件优先,按编号排序。
    '''
    models = []

    if not CLASSIFY_MODEL_ROOT.exists():
        return models

    seen_ids = set()

    # 根目录平铺的模型文件: 1.tflite -> id=1
    for tflite_path in sorted(CLASSIFY_MODEL_ROOT.glob('*.tflite')):
        model_id = tflite_path.stem
        if model_id in seen_ids:
            continue
        seen_ids.add(model_id)
        models.append({
            'id': model_id,
            'name': f'分类模型 {model_id}',
            'tflite_file': tflite_path.name,
        })

    # 子目录模型: models/classify/<id>/xxx.tflite
    for item in CLASSIFY_MODEL_ROOT.iterdir():
        if not item.is_dir():
            continue

        tflite_path = _find_tflite_path(item)
        if tflite_path is None:
            continue

        if item.name in seen_ids:
            continue
        seen_ids.add(item.name)

        models.append({
            'id': item.name,
            'name': f'分类模型 {item.name}',
            'tflite_file': tflite_path.name,
        })

    def sort_key(model):
        model_id = model['id']
        return (0, int(model_id)) if model_id.isdigit() else (1, model_id)

    models.sort(key=sort_key)
    return models
