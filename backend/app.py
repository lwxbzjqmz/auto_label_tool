from pathlib import Path
from flask import Flask,jsonify,send_from_directory,abort,request
from config import FRONTEND_DIR,IMAGE_DIR,ANNOTATION_DIR
from image_manager import list_images
from xml_writer import write_voc_xml

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
    '''返回图片列表'''
    images=list_images()
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

    app.run(host="127.0.0.1",port=5000,debug=True)
