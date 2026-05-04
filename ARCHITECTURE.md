把下面内容保存成：

```text
ARCHITECTURE.md
```

放在项目根目录：

```text
W:\openmv-ide\object_file\auto_label_tool\ARCHITECTURE.md
```

内容如下：

````markdown
# Auto Label Tool 项目架构说明

## 1. 项目目标

本项目是一个本地半自动标注工具，用于“二十一届蚂蚁搬家”目标检测数据集制作。

核心目标：

```text
图片输入
→ 模型或人工得到若干组 boxes
→ 前端显示矩形框
→ 用户可以修改、删除、补框
→ 后端保存为 VOC XML
````

统一坐标格式：

```json
[
  [xmin, ymin, xmax, ymax],
  [xmin, ymin, xmax, ymax]
]
```

当前任务只做目标检测，不做分类。

虽然真实目标有多个类别，但训练阶段统一写成：

```xml
<name>object</name>
```

---

## 2. 当前开发环境

```text
浏览器：Microsoft Edge
编辑器：VS Code
环境管理：Anaconda
Conda 环境名：autolabel
版本控制：Git
后端：Python + Flask
前端：HTML + CSS + JavaScript + Canvas
```

当前已安装的主要 Python 库：

```text
Flask
numpy
opencv-python
pillow
tensorflow
```

---

## 3. 当前项目目录结构

```text
auto_label_tool/
├── backend/
│   ├── app.py
│   ├── config.py
│   ├── detector.py
│   ├── image_manager.py
│   ├── xml_reader.py
│   ├── xml_writer.py
│   └── debug/
│       ├── debug_yolo_output.py
│       └── inspect_tflite.py
│
├── frontend/
│   ├── index.html
│   ├── main.js
│   └── style.css
│
├── data/
│   ├── JPEGImages/
│   └── Annotations/
│
├── models/
│   ├── yolov3_mbv2/
│   │   └── 1/
│   │       ├── yolo3_iou_smartcar_final.tflite
│   │       └── yolo3_anchors.txt
│   │
│   └── large_model/
│
├── .gitignore
└── ARCHITECTURE.md
```

---

## 4. 各目录作用

### backend/

后端代码目录。

负责：

```text
读取图片列表
提供图片访问接口
读取已有 XML
保存 VOC XML
调用模型返回 boxes
```

主要文件：

```text
app.py
    Flask 后端入口，提供 API 接口。

config.py
    全局配置文件，包括数据路径、模型路径、检测参数。

detector.py
    自动标注统一入口。
    get_boxes(image_path) 是核心函数。

image_manager.py
    管理 data/JPEGImages 下的图片列表。

xml_reader.py
    读取已有 VOC XML，并转换成 boxes。

xml_writer.py
    把 boxes 保存成 VOC XML。

debug/
    调试脚本目录，不属于正式业务链路。
```

---

### frontend/

前端代码目录。

负责：

```text
显示图片
Canvas 画框
手动新增框
删除框
清空框
显示 boxes 坐标
点击自动标注
点击保存 XML
```

主要文件：

```text
index.html
    页面结构。

style.css
    页面样式。

main.js
    前端核心逻辑，包括图片显示、Canvas 画框、请求后端接口。
```

---

### data/

数据目录。

```text
data/JPEGImages/
    存放 .jpg 图片。

data/Annotations/
    存放生成的 .xml 标注文件。
```

当前项目只处理：

```text
.jpg
```

---

### models/

模型目录。

```text
models/yolov3_mbv2/
    存放自己的 YOLOv3-mbv2 模型。

models/large_model/
    后续用于大模型相关文件、协议、批量结果等。
```

当前本地模型目录示例：

```text
models/yolov3_mbv2/1/
├── yolo3_iou_smartcar_final.tflite
└── yolo3_anchors.txt
```

注意：

```text
yolo3_iou_smartcar_final.tflite
    是 PC 端自动标注使用的模型。

yolo3_iou_smartcar_final_with_post_processing.tflite
    是 OpenMV / OpenART 部署端模型，不放入当前 PC 推理链路。
```

---

## 5. config.py 当前约定

当前 config.py 的核心变量如下。

```python
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

FRONTEND_DIR = BASE_DIR / "frontend"

DATA_DIR = BASE_DIR / "data"

IMAGE_DIR = DATA_DIR / "JPEGImages"

ANNOTATION_DIR = DATA_DIR / "Annotations"

IMAGE_EXTENSIONS = {".jpg"}

CLASS_NAME = "object"

VOC_FOLDER_NAME = "detect.class"

MODEL_DIR = BASE_DIR / "models"

MODEL_SELECT = 1

YOLOV3_MBV2_PC_MODEL_PATH = (
    MODEL_DIR
    / "yolov3_mbv2"
    / str(MODEL_SELECT)
    / "yolo3_iou_smartcar_final.tflite"
)

YOLOV3_MBV2_ANCHORS_PATH = (
    MODEL_DIR
    / "yolov3_mbv2"
    / str(MODEL_SELECT)
    / "yolo3_anchors.txt"
)
```

当前不使用以下变量名：

```text
YOLOV3_MBV2_MODEL_NAME
YOLOV3_MBV2_OPENMV_MODEL_PATH
```

原因：

```text
本工程只关心 PC 端自动标注模型。
OpenMV / OpenART 部署模型不进入当前 PC 后端推理链路。
```

---

## 6. YOLOv3-mbv2 检测配置

YOLOv3-mbv2 PC 端模型当前参数：

```text
模型输入尺寸：144 × 144 × 3
模型输入 dtype：int8
模型输出尺寸：18 × 18 × 18
模型输出 dtype：int8
anchors：4,4, 6,6, 19,18
```

anchors 文件内容：

```text
4,4, 6,6, 19,18
```

每个模型目录下都应该有自己的 anchors 文件：

```text
models/yolov3_mbv2/1/yolo3_anchors.txt
models/yolov3_mbv2/2/yolo3_anchors.txt
models/yolov3_mbv2/3/yolo3_anchors.txt
```

即使不同模型使用同一份 anchors，也建议复制一份放到对应模型目录，避免混乱。

---

## 7. YOLOv3-mbv2 图像预处理约定

原始图片尺寸：

```text
320 × 240
```

检测前裁剪区域：

```text
x = 40
y = 0
w = 240
h = 240
```

也就是从原图中裁剪中间的：

```text
240 × 240
```

PIL 裁剪写法：

```python
img.crop((40, 0, 280, 240))
```

然后 resize 到：

```text
144 × 144
```

再输入 TFLite 模型。

模型输出框需要反向映射：

```text
144×144 模型坐标
→ 240×240 裁剪图坐标
→ 加回 CROP_X = 40
→ 原始 320×240 图片坐标
```

---

## 8. detector.py 设计原则

detector.py 的唯一正式入口：

```python
get_boxes(image_path)
```

返回格式：

```python
[
    [xmin, ymin, xmax, ymax],
    [xmin, ymin, xmax, ymax]
]
```

模型选择逻辑：

```text
MODEL_SELECT = 0
    后续走 large_model 路线。

MODEL_SELECT != 0
    走 yolov3_mbv2 PC 端 TFLite 模型。
```

当前已经完成：

```text
YOLOv3-mbv2 PC 端 TFLite 推理
YOLO 输出解码
NMS 去重
boxes 返回
前端自动标注显示 boxes
```

当前不再使用假坐标作为正式流程。

---

## 9. 当前已完成的功能

当前项目已完成：

```text
1. Flask 后端启动
2. Edge 打开前端页面
3. 前端读取图片列表
4. 前端显示图片
5. Canvas 手动画框
6. 坐标列表显示 boxes
7. 删除选中框
8. 清空所有框
9. 前端 boxes 发送给后端
10. 后端保存 VOC XML
11. 后端读取已有 VOC XML
12. 打开图片时回显已有框
13. 假坐标自动标注链路测试
14. PC 端 YOLOv3-mbv2 TFLite 模型接入
15. 真实模型自动标注已经能返回 boxes
```

---

## 10. VOC XML 输出格式

XML 按用户提供的模板生成。

基本结构：

```xml
<annotation>
    <folder>detect.class</folder>
    <filename>00000.jpg</filename>
    <path>...</path>
    <source>
        <database>Unknown</database>
    </source>
    <size>
        <width>320</width>
        <height>240</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>
    <object>
        <name>object</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>...</xmin>
            <ymin>...</ymin>
            <xmax>...</xmax>
            <ymax>...</ymax>
        </bndbox>
    </object>
</annotation>
```

一组 box 对应一个：

```xml
<object>
    ...
</object>
```

所有 object 的 name 固定为：

```xml
<name>object</name>
```

---

## 11. 前后端 API 约定

### GET /api/images

获取图片列表。

返回示例：

```json
{
  "success": true,
  "images": ["00000.jpg", "00001.jpg"],
  "count": 2
}
```

---

### GET /api/image/<image_name>

返回指定图片。

---

### GET /api/annotation/<image_name>

读取已有 XML，返回 boxes。

返回示例：

```json
{
  "success": true,
  "result": {
    "exists": true,
    "xml_name": "00000.xml",
    "boxes": [
      [120, 50, 180, 100]
    ]
  }
}
```

---

### POST /api/detect

自动标注接口。

请求示例：

```json
{
  "image_name": "00000.jpg"
}
```

返回示例：

```json
{
  "success": true,
  "message": "Detect success",
  "boxes": [
    [120, 50, 180, 100],
    [200, 70, 250, 130]
  ],
  "box_count": 2
}
```

---

### POST /api/save

保存 XML。

请求示例：

```json
{
  "image_name": "00000.jpg",
  "boxes": [
    [120, 50, 180, 100],
    [200, 70, 250, 130]
  ]
}
```

返回示例：

```json
{
  "success": true,
  "message": "XML saved successfully",
  "result": {
    "xml_name": "00000.xml",
    "box_count": 2
  }
}
```

---

## 12. large_model 后续规划

large_model 是下一阶段大工程。

当前只是预留目录：

```text
models/large_model/
```

large_model 的定位：

```text
不是替代当前 yolov3_mbv2 单张自动标注链路。
而是新增一条批量自动标注路线。
```

计划目标：

```text
读取 data/JPEGImages 下所有 .jpg
→ 批量发送给大模型或按协议分批发送
→ 大模型返回包含 boxes 和 name 的 JSON
→ Python 解析 JSON
→ 批量生成 XML
→ 前端负责查看和修正
```

预期 JSON 格式：

```json
{
  "images": [
    {
      "image_name": "00000.jpg",
      "objects": [
        {
          "name": "object",
          "box": [120, 50, 180, 100]
        }
      ]
    }
  ]
}
```

虽然大模型可能返回 name，但当前训练 XML 仍然可以统一写：

```xml
<name>object</name>
```

除非后续明确要做分类检测。

large_model 后续建议目录结构：

```text
backend/
└── large_model/
    ├── client.py
    ├── request_builder.py
    ├── response_parser.py
    ├── batch_runner.py
    └── schema.py
```

各文件作用：

```text
client.py
    负责和大模型 API 通信。

request_builder.py
    负责整理图片、构造请求数据。

response_parser.py
    负责解析大模型返回 JSON，转成统一 boxes 格式。

batch_runner.py
    负责批量运行 large_model 标注流程。

schema.py
    负责约定 large_model 的输入输出格式。
```

large_model 不应该直接写进 detector.py 的 YOLO 解码逻辑中。

detector.py 最多只作为统一入口：

```python
if MODEL_SELECT == 0:
    return get_boxes_by_large_model(image_path)
```

但批量 large_model 更适合单独 API 或单独脚本。

---

## 13. 当前重要原则

### 原则 1：前端只认 boxes

前端不关心 boxes 来自哪里。

```text
YOLOv3-mbv2
large_model
人工画框
```

最终都必须转换成：

```json
[
  [xmin, ymin, xmax, ymax]
]
```

---

### 原则 2：XML 只由后端生成

前端不直接生成 XML。

前端只发送：

```json
{
  "image_name": "...",
  "boxes": [...]
}
```

后端负责生成 XML。

---

### 原则 3：模型文件和 anchors 绑定

每个模型目录下都应放：

```text
模型文件
anchors 文件
```

例如：

```text
models/yolov3_mbv2/1/
├── yolo3_iou_smartcar_final.tflite
└── yolo3_anchors.txt
```

---

### 原则 4：不靠猜继续开发

如果后续上下文不清楚，需要重新确认这些文件：

```text
ARCHITECTURE.md
backend/config.py
backend/detector.py
backend/app.py
frontend/main.js
```

尤其进入 large_model 阶段后，如果结构变化，必须先同步最新结构。

---

## 14. 常用命令

启动后端：

```powershell
cd W:\openmv-ide\object_file\auto_label_tool
conda activate autolabel
python .\backend\app.py
```

Edge 打开：

```text
http://127.0.0.1:5000
```

查看项目结构：

```powershell
Get-ChildItem -Recurse -Name
```

测试 detect 接口：

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:5000/api/detect" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"image_name":"00000.jpg"}'
```

测试 save 接口：

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:5000/api/save" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"image_name":"00000.jpg","boxes":[[10,20,80,90]]}'
```

Git 存档：

```powershell
git status
git add .
git commit -m "message"
```

---

## 15. 当前阶段结论

当前阶段已经完成：

```text
YOLOv3-mbv2 PC 端自动标注主链路
```

当前稳定目标：

```text
前端选择图片
→ 点击自动标注
→ 后端 yolov3_mbv2 返回真实 boxes
→ 前端显示框
→ 人工修正
→ 保存 XML
```

下一阶段：

```text
large_model 批量标注方案设计与接入
```

```
```
