from config import DEFAULT_CLASSIFIER,CLASSIFIER_PURE_COLOR,CLASSIFIER_MODEL


def classify_class_boxes(image_path,class_boxes,rules=None,classifier=None,model_select=None):
    '''统一分类入口'''
    classifier = classifier or DEFAULT_CLASSIFIER

    if classifier == CLASSIFIER_PURE_COLOR:
        from pure_color_classify.classify import classify_class_boxes as pure_color_classify
        return pure_color_classify(image_path,class_boxes,rules=rules)

    if classifier == CLASSIFIER_MODEL:
        from model_classify.classify import classify_class_boxes as model_classify
        return model_classify(image_path,class_boxes,model_select=model_select)

    raise ValueError(f"Unknown classifier:{classifier}")
