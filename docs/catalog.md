[TOC]

# auto_label_tool

---

## backend

### backend 根目录文件

- `app.py` - 服务器启动入口
- `config.py` - 全局配置入口
- `detector.py` - 模型选择
- `grounding_dino_manager.py`- grounding_dino加载模型必要函数
- `image_manager.py` - 读取data所有的图片文件名
- `start_grounding_dino.ps1` - 手动加载grounding_dino的powershell脚本
- `xml_reader.py` - voc xml 转 boxes
- `xml_writer.py` - boxes 转 voc xml

### backend/large_model

#### backend/large_model 根目录文件

- `batch_runner.py` - 将批量json，转化为voc xml,项目中目前没有使用
- `detector.py` - 封装函数，供上一级detector.py使用，项目中用处不大
- `response_parser.py` - 读取批量json
- `schema.py` - large_model 返回的JOSN协议
- `__init__.py` - 由于使用生成式在测试时使用效果一般，这一部分基本不使用

##### backend/large_model/grounding_dino

- `client.py` - 请求Grounding DINO worker，返回boxes
- `worker.py` - Grounding DINO worker服务器的搭建
- `__init__.py`

### backend/yolov3_mbv2

#### backend/yolov3_mbv2 根目录文件

- `anchors.py` - 读取anchors.txt
- `detector.py` - 调用yolo模型的总入口
- `postprocess.py` - 解码yolo输出，并处理NMS，IoU，返回裁剪前的坐标
- `preprocess.py` - 对图片进行裁剪和归一化的处理，规则取决于训模型时的参数
- `__init__.py`

---

## data

### data/Annotations

### data/JPEGImages

---

## docs

- `ARCHITECTURE.md`
- `bug记录.txt`
- `learn_markdown.md`
- `初步架构.txt`
- `标注工具可行性.txt`

---

## experiments

### experiments/debug

- `debug_yolo_output.py`
- `inspect_tflite.py`

### experiments/grounding_dino

- `test_one_image.py`

---

## frontend

- `index.html`
- `main.js`
- `style.css`

---

## models

### models/large_model

#### models/large_model/responses

- `large_model_result.json`

### models/yolov3_mbv2

#### models/yolov3_mbv2/1

- `yolo3_anchors.txt`
- `yolo3_iou_smartcar_final.tflite`

#### models/yolov3_mbv2/2

- `yolo3_anchors.txt`
- `yolo3_iou_smartcar_final.tflite`