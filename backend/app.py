from pathlib import Path
from pprint import pformat
import json
import importlib
from flask import Flask,jsonify,send_from_directory,abort,request,send_file
from config import FRONTEND_DIR,IMAGE_DIR,ANNOTATION_DIR,YOLOV3_MODEL_SELECT,DEFAULT_DETECTOR
import config as runtime_config
from image_manager import list_images
from xml_writer import write_voc_xml
import xml_writer as xml_writer_module
from xml_reader import read_voc_xml
from detector import get_detect_result
from classify import classify_class_boxes
import pure_color_classify.classify as pure_color_classify_module
from grounding_dino_manager import start_grounding_dino_worker,stop_grounding_dino_worker
from yolov3_mbv2.model_paths import list_yolov3_mbv2_models
from model_classify.model_paths import list_classify_models
from backup_manager import make_annotations_backup
from reject_manager import reject_image

app=Flask(__name__)
SHORTCUT_CONFIG_PATH=Path(__file__).resolve().parent/"shortcut_config.json"

@app.route("/")
def index():
    '''打开前端网页'''
    return send_from_directory(FRONTEND_DIR,"index.html")

@app.route("/frontend/<path:filename>")
def frontend_files(filename):
    '''提供前端CSS/JS文件'''
    return send_from_directory(FRONTEND_DIR,filename)

@app.route("/api/images")
def api_images():
    '''返回图片列表,包含标注框信息'''
    image_names=list_images()
    images=[]
    for name in image_names:
        annotation=read_voc_xml(name)
        images.append({
            "name":name,
            "has_annotation":annotation["exists"],
            "class_boxes":annotation["class_boxes"]
        })
    return jsonify({
        "success" : True,
        "images" : images,
        "count" : len(images)
    })

@app.route("/api/image/<path:image_name>")
def api_image(image_name):
    '''前端返回指定图片'''
    image_path=(IMAGE_DIR/image_name).resolve()

#防止路径穿越
    try:
        image_path.relative_to(IMAGE_DIR.resolve())
    except ValueError:
        abort(400,"Invalid image path")
    if not image_path.exists() or not image_path.is_file():
        abort(404,"Image not found")

    return send_from_directory(IMAGE_DIR,image_name)

@app.route("/api/save",methods=["POST"])
def api_save():
    '''接收前端boxes,生成voc xml'''
    data=request.get_json(silent=True)

    if not data:
        return jsonify({
            "success":False,
            "message":"No JSON data received"
        }),400
    
    image_name=data.get("image_name")
    class_boxes=data.get("class_boxes",[])

    if not image_name:
        return jsonify({
            "success":False,
            "message":"Missing image_name"
        }),400
    
    if not isinstance(class_boxes,list):
        return jsonify({
            "success":False,
            "message":"class_boxes must be a list"
        }),400
    
    try:
        result=write_voc_xml(image_name,class_boxes)
    except FileExistsError as e:
        return jsonify({
            "success":False,
            "message":str(e)
        }),404
    except Exception as e:
        return jsonify({
            "success":False,
            "message":f"Failed to save XML:{e}"
        }),500
    
    return jsonify({
        "success":True,
        "message":"XML saved successfully",
        "result":result
        })


@app.route("/api/shortcut_config")
def api_shortcut_config():
    '''return saved shortcut config'''
    return jsonify({
        "success":True,
        "config":_read_shortcut_config()
    })


@app.route("/api/shortcut_config",methods=["POST"])
def api_save_shortcut_config():
    '''save shortcut config'''
    data=request.get_json(silent=True)

    if not data:
        return jsonify({
            "success":False,
            "message":"No JSON data received"
        }),400

    config=data.get("config")

    if not isinstance(config,dict):
        return jsonify({
            "success":False,
            "message":"config must be an object"
        }),400

    try:
        config=_normalize_shortcut_config(config)
        _write_shortcut_config(config)
    except ValueError as e:
        return jsonify({
            "success":False,
            "message":str(e)
        }),400
    except Exception as e:
        return jsonify({
            "success":False,
            "message":f"Save shortcut config failed:{e}"
        }),500

    return jsonify({
        "success":True,
        "message":"Shortcut config saved",
        "config":config
    })


@app.route("/api/reject_image",methods=["POST"])
def api_reject_image():
    '''把不合格图片和对应XML移入废弃区'''
    data=request.get_json(silent=True)

    if not data:
        return jsonify({
            "success":False,
            "message":"No JSON data received"
        }),400
    
    image_name=data.get("image_name")

    if not image_name:
        return jsonify({
            "success":False,
            "message":"Missing image_name"
        }),400
    
    try:
        result=reject_image(image_name)
    except ValueError as e:
        return jsonify({
            "success":False,
            "message":str(e)
        }),400
    except FileNotFoundError as e:
        return jsonify({
            "success":False,
            "message":str(e)
        }),404
    except Exception as e:
        return jsonify({
            "success":False,
            "message":f"Reject image failed:{e}"
        }),500
    
    return jsonify({
        "success":True,
        "message":"Image moved to rejected folder",
        "result":result
    })

@app.route("/api/annotation/<path:image_name>")
def api_annontation(image_name):
    '''读voc xml,返回boxes'''
    try:
        result=read_voc_xml(image_name)
    except ValueError as e:
        return jsonify({
            "success":False,
            "message":str(e)
        }),500
    except Exception as e:
        return jsonify({
            "success":False,
            "message":f"Failed to read XML:{e}"
        }),500
    
    return jsonify({
        "success":True,
        "result":result
    })

@app.route("/api/detect",methods=["POST"])
def api_detect():
    '''自动标注接口'''
    data=request.get_json(silent=True)

    if not data:
        return jsonify({
            "success":False,
            "message":"No JOSN data received"
        }),400
    
    image_name=data.get("image_name")
    detector=data.get("detector",DEFAULT_DETECTOR)
    model_select=data.get("model_select",None)
    if model_select is None or model_select == "":
        model_select = runtime_config.CLASSIFY_MODEL_SELECT

    grounding_dino_options={
        "prompt":data.get("prompt"),
        "box_threshold":data.get("box_threshold"),
        "text_threshold":data.get("text_threshold"),
        "use_nms":data.get("use_nms"),
        "nms_iou_threshold":data.get("nms_iou_threshold"),
    }

    if not image_name:
        return jsonify({
            "success":False,
            "message":"Missing image_name"
        }),400
    
    image_path=(IMAGE_DIR/image_name).resolve()

    try:
        image_path.relative_to(IMAGE_DIR.resolve())
    except ValueError:
        return jsonify({
            "success":False,
            "message":"Invalid image path"
        }),400
    
    if not image_path.exists() or not image_path.is_file():
        return jsonify({
            "success":False,
            "message":f"Image not found:{image_name}"
        }),404
    
    try:
        detect_result=get_detect_result(
            image_path,
            detector=detector,
            model_select=model_select,
            grounding_dino_options=grounding_dino_options
        )

        class_boxes=detect_result.get("class_boxes",[])
        objects=detect_result.get("objects",[])

    except Exception as e:
        return jsonify({
            "success":False,
            "message":f"Detect failed:{e}"
        }),500

    return jsonify({
        "success":True,
        "message":"Detect success",
        "class_boxes":class_boxes,
        "objects":objects,
        "box_count":len(class_boxes)
    })


@app.route("/api/classify",methods=["POST"])
def api_classify():
    '''根据当前class_boxes进行分类'''
    data=request.get_json(silent=True)

    if not data:
        return jsonify({
            "success":False,
            "message":"No JSON data received"
        }),400

    image_name=data.get("image_name")
    class_boxes=data.get("class_boxes",[])
    rules=data.get("rules")
    classifier=data.get("classifier") or runtime_config.DEFAULT_CLASSIFIER
    model_select=data.get("model_select") or runtime_config.CLASSIFY_MODEL_SELECT

    if not image_name:
        return jsonify({
            "success":False,
            "message":"Missing image_name"
        }),400

    if not isinstance(class_boxes,list):
        return jsonify({
            "success":False,
            "message":"class_boxes must be a list"
        }),400

    if rules is not None:
        if not isinstance(rules,list):
            return jsonify({
                "success":False,
                "message":"rules must be a list"
            }),400
        try:
            rules=_normalize_classify_rules(rules)
        except ValueError as e:
            return jsonify({
                "success":False,
                "message":str(e)
            }),400

    image_path=(IMAGE_DIR/image_name).resolve()

    try:
        image_path.relative_to(IMAGE_DIR.resolve())
    except ValueError:
        return jsonify({
            "success":False,
            "message":"Invalid image path"
        }),400

    if not image_path.exists() or not image_path.is_file():
        return jsonify({
            "success":False,
            "message":f"Image not found:{image_name}"
        }),404

    try:
        class_boxes=classify_class_boxes(
            image_path,
            class_boxes,
            rules=rules,
            classifier=classifier,
            model_select=model_select
        )
    except Exception as e:
        return jsonify({
            "success":False,
            "message":f"Classify failed:{e}"
        }),500

    return jsonify({
        "success":True,
        "message":"Classify success",
        "class_boxes":class_boxes,
        "box_count":len(class_boxes)
    })



@app.route("/api/classify_rules")
def api_classify_rules():
    '''return classify RGB rules'''
    _reload_runtime_config()

    return jsonify({
        "success":True,
        "default_class_name":runtime_config.CLASSIFY_DEFAULT_CLASS_NAME,
        "center_ratio":runtime_config.CLASSIFY_CENTER_RATIO,
        "rules":runtime_config.CLASSIFY_PIXEL_RULES
    })


@app.route("/api/classify_rules",methods=["POST"])
def api_save_classify_rules():
    '''save classify RGB rules to config.py'''
    data=request.get_json(silent=True)

    if not data:
        return jsonify({
            "success":False,
            "message":"No JSON data received"
        }),400

    rules=data.get("rules",[])

    if not isinstance(rules,list):
        return jsonify({
            "success":False,
            "message":"rules must be a list"
        }),400

    try:
        rules=_normalize_classify_rules(rules)
        _write_classify_rules_to_config(rules)
    except ValueError as e:
        return jsonify({
            "success":False,
            "message":str(e)
        }),400
    except Exception as e:
        return jsonify({
            "success":False,
            "message":f"Save classify rules failed:{e}"
        }),500

    runtime_config.CLASSIFY_PIXEL_RULES=rules
    pure_color_classify_module.CLASSIFY_PIXEL_RULES=rules

    class_names=_get_class_names_by_rules(rules)
    runtime_config.CLASS_NAME=class_names
    xml_writer_module.CLASS_NAME=class_names

    return jsonify({
        "success":True,
        "message":"Classify rules saved",
        "rules":rules,
        "class_names":class_names
    })


def _normalize_classify_rules(rules):
    next_rules=[]

    for index,rule in enumerate(rules):
        if not isinstance(rule,dict):
            raise ValueError(f"rule {index + 1} must be an object")

        name=str(rule.get("name","")).strip()
        if not name:
            raise ValueError(f"rule {index + 1} missing name")

        rgb_min=_normalize_rgb(rule.get("rgb_min"),f"{name}.rgb_min")
        rgb_max=_normalize_rgb(rule.get("rgb_max"),f"{name}.rgb_max")

        for min_value,max_value in zip(rgb_min,rgb_max):
            if min_value>max_value:
                raise ValueError(f"{name} rgb_min must <= rgb_max")

        next_rules.append({
            "name":name,
            "rgb_min":rgb_min,
            "rgb_max":rgb_max,
        })

    return next_rules


def _normalize_rgb(value,field_name):
    if not isinstance(value,(list,tuple)) or len(value)!=3:
        raise ValueError(f"{field_name} must have 3 values")

    rgb=[]
    for item in value:
        item=int(item)
        if item<0 or item>255:
            raise ValueError(f"{field_name} value must be 0-255")
        rgb.append(item)

    return tuple(rgb)


def _get_class_names_by_rules(rules):
    class_names=[runtime_config.CLASSIFY_DEFAULT_CLASS_NAME]

    for rule in rules:
        name=rule["name"]
        if name not in class_names:
            class_names.append(name)

    return class_names


def _write_classify_rules_to_config(rules):
    config_path=Path(__file__).resolve().parent/"config.py"
    text=config_path.read_text(encoding="utf-8")

    class_names=_get_class_names_by_rules(rules)
    text=_replace_config_assignment(text,"CLASS_NAME",pformat(class_names, width=120))
    text=_replace_config_assignment(text,"CLASSIFY_PIXEL_RULES",pformat(rules, width=120))

    config_path.write_text(text,encoding="utf-8")


def _replace_config_assignment(text,name,value_text):
    start=text.find(f"{name}=")
    if start<0:
        raise ValueError(f"Missing config item:{name}")

    value_start=start+len(name)+1
    index=value_start

    while index<len(text) and text[index].isspace() and text[index] not in "\r\n":
        index+=1

    if index>=len(text):
        raise ValueError(f"Invalid config item:{name}")

    if text[index] in "[({":
        pairs={"[":"]","(":")","{":"}"}
        stack=[]
        in_string=False
        string_char=""
        escape=False
        end=index

        while end<len(text):
            char=text[end]

            if in_string:
                if escape:
                    escape=False
                elif char=="\\":
                    escape=True
                elif char==string_char:
                    in_string=False
            else:
                if char in "'\"":
                    in_string=True
                    string_char=char
                elif char in pairs:
                    stack.append(pairs[char])
                elif stack and char==stack[-1]:
                    stack.pop()
                    if not stack:
                        end+=1
                        break
            end+=1
    else:
        end=text.find("\n",index)
        if end<0:
            end=len(text)

    return text[:value_start]+value_text+text[end:]


def _reload_runtime_config():
    importlib.reload(runtime_config)
    pure_color_classify_module.CLASSIFY_PIXEL_RULES=runtime_config.CLASSIFY_PIXEL_RULES
    xml_writer_module.CLASS_NAME=runtime_config.CLASS_NAME


def _read_shortcut_config():
    if not SHORTCUT_CONFIG_PATH.exists():
        return {}

    with SHORTCUT_CONFIG_PATH.open("r",encoding="utf-8") as f:
        data=json.load(f)

    if not isinstance(data,dict):
        return {}

    return _normalize_shortcut_config(data)


def _write_shortcut_config(config):
    SHORTCUT_CONFIG_PATH.write_text(
        json.dumps(config,ensure_ascii=False,indent=4),
        encoding="utf-8"
    )


def _normalize_shortcut_config(config):
    allowed_keys={
        "previousImage",
        "nextImage",
        "saveXml",
        "yoloDetect",
        "groundingDinoDetect",
        "classifyPureColor",
        "classifyModel",
        "deleteBox",
        "clearBoxes",
        "reloadAnnotation",
        "rejectImage",
        "cancelSelect",
    }

    next_config={}

    for key in allowed_keys:
        value=config.get(key)
        if value is None:
            continue
        value=str(value).strip().lower()
        # 空字符串表示"该功能无快捷键"(前端按 Backspace 清除),也一并保存
        next_config[key]=value

    return next_config


@app.route("/api/yolo_models")
def api_yolo_models():
    '''返回YOLO模型列表'''
    models=list_yolov3_mbv2_models()

    return jsonify({
        "success":True,
        "models":models,
        "default_model":YOLOV3_MODEL_SELECT
    })


@app.route("/api/classify_models")
def api_classify_models():
    '''返回分类模型列表'''
    models=list_classify_models()

    return jsonify({
        "success":True,
        "models":models,
        "default_model":runtime_config.CLASSIFY_MODEL_SELECT,
        "classifiers":[
            {
                "id":runtime_config.CLASSIFIER_PURE_COLOR,
                "name":"纯色分类"
            },
            {
                "id":runtime_config.CLASSIFIER_MODEL,
                "name":"模型分类"
            }
        ]
    })
@app.route("/api/detectors")
def api_detectors():
    '''返回检测器信息'''
    return jsonify({
        "success":True,
        "default_detector":DEFAULT_DETECTOR,
        "detectors":[
            {
                "id":"yolo",
                "name":"YOLOv3-mbv2"
            },
            {
                "id":"grounding_dino",
                "name":"Grounding DINO"
            }
        ]
    })

@app.route("/api/grounding_dino_defaults")
def api_grounding_dino_defaults():
    """返回 Grounding DINO 默认配置"""
    from config import (
        GROUNDING_DINO_DEFAULT_PROMPT,
        GROUNDING_DINO_BOX_THRESHOLD,
        GROUNDING_DINO_TEXT_THRESHOLD,
        GROUNDING_DINO_USE_NMS,
        GROUNDING_DINO_NMS_IOU_THRESHOLD,
    )
    return jsonify({
        "success": True,
        "defaults": {
            "prompt": GROUNDING_DINO_DEFAULT_PROMPT,
            "box_threshold": GROUNDING_DINO_BOX_THRESHOLD,
            "text_threshold": GROUNDING_DINO_TEXT_THRESHOLD,
            "use_nms": GROUNDING_DINO_USE_NMS,
            "nms_iou_threshold": GROUNDING_DINO_NMS_IOU_THRESHOLD,
        }
    })

@app.route("/api/backup_annotations")
def api_backup_annotations():
    '''备份Annotations目录'''
    try:
        result=make_annotations_backup()
    except FileNotFoundError as e:
        return jsonify({
            "success":False,
            "message":str(e)
        }),404
    except Exception as e:
        return jsonify({
            "success":False,
            "message":f"Backup failed:{e}"
        }),500
    
    return send_file(
        result["zip_path"],
        as_attachment=True,
        download_name=result["zip_name"],
        mimetype="application/zip"
    )

@app.route("/api/health")
def api_health():
    '''测试后端是否正常'''
    return jsonify({
        "success":True,
        "message":"auto label backend is running"
    })

if __name__=="__main__":
    IMAGE_DIR.mkdir(parents=True,exist_ok=True)
    ANNOTATION_DIR.mkdir(parents=True,exist_ok=True)

    start_grounding_dino_worker()

    try:
        app.run(host="127.0.0.1",port=5000,debug=True,use_reloader=False)
    finally:
        stop_grounding_dino_worker()






