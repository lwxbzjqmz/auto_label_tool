from pathlib import Path
from flask import Flask,jsonify,send_from_directory,abort,request,send_file
from config import FRONTEND_DIR,IMAGE_DIR,ANNOTATION_DIR,MODEL_SELECT,DEFAULT_DETECTOR
from image_manager import list_images
from xml_writer import write_voc_xml
from xml_reader import read_voc_xml
from detector import get_detect_result
from grounding_dino_manager import start_grounding_dino_worker,stop_grounding_dino_worker
from yolov3_mbv2.model_paths import list_yolov3_mbv2_models
from backup_manager import make_annotations_backup
from reject_manager import reject_image

app=Flask(__name__)

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
            "boxes":annotation["boxes"]
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
    boxes=data.get("boxes",[])

    if not image_name:
        return jsonify({
            "success":False,
            "message":"Missing image_name"
        }),400
    
    if not isinstance(boxes,list):
        return jsonify({
            "success":False,
            "message":"boxes must be a list"
        }),400
    
    try:
        result=write_voc_xml(image_name,boxes)
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

        boxes=detect_result.get("boxes",[])
        objects=detect_result.get("objects",[])

    except Exception as e:
        return jsonify({
            "success":False,
            "message":f"Detect failed:{e}"
        }),500

    return jsonify({
        "success":True,
        "message":"Detect success",
        "boxes":boxes,
        "objects":objects,
        "box_count":len(boxes)
    })

@app.route("/api/yolo_models")
def api_yolo_models():
    '''返回YOLO模型列表'''
    models=list_yolov3_mbv2_models()

    return jsonify({
        "success":True,
        "models":models,
        "default_model":MODEL_SELECT
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

