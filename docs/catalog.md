
# auto_label_tool的目录

---

## 根目录

- `CLAUDE.md`  
  - 项目说明(供 AI 助手参考)
- `environment_autolabel.yml`  
  - autolabel conda 环境定义(Flask + YOLO 推理)
- `environment_grounding_dino.yml`  
  - grounding_dino conda 环境定义(worker 进程)
- `backend/`、`frontend/`、`data/`、`models/`、`experiments/`、`docs/`

---

## backend

### backend 根目录文件

- `app.py`  
  - Flask 服务器启动入口,提供 REST API
- `config.py`  
  - 全局配置入口(路径、超参数、模型选择)
- `detector.py`  
  - 检测器选择(yolo / grounding_dino),统一返回 `{boxes, objects}`
- `classify.py`  
  - 分类器统一入口(纯色分类 / 模型分类)
- `grounding_dino_manager.py`  
  - Grounding DINO worker 子进程的启动/停止管理
- `image_manager.py`  
  - 读取data/JPEGImages所有的图片文件名
- `xml_reader.py`  
  - voc xml 转 boxes
- `xml_writer.py`  
  - boxes 转 voc xml
- `reject_manager.py`  
  - 把不满足的图片连同xml放入废弃区  
- `backup_manager.py`  
  - 把 Annotations 目录打包成 zip 备份
- `batch_detector_and_save.py`  
  - 自定义批量检测并保存标注(独立脚本,不在前端流程内)
- `batch_model_classify_and_save.py`  
  - 批量模型分类并保存标注(独立脚本,用 model_classify 的 tflite 模型)
- `run.ps1`  
  - Windows 启动辅助脚本(检查路径后使用)
- `shortcut_config.json`  
  - 快捷键配置,由前端"保存快捷键"写入(运行时生成)

### backend/grounding_dino  

#### backend/grounding_dino  根目录文件

- `client.py` - 请求Grounding DINO worker，返回boxes
- `detector.py` - grounding_dino 检测的封装
- `worker.py` - Grounding DINO worker服务器的搭建(独立进程,PyTorch,端口 15051)
- `__init__.py`

### backend/yolov3_mbv2

#### backend/yolov3_mbv2 根目录文件

- `anchors.py` - 读取anchors.txt
- `detector.py` - 调用yolo模型的总入口
- `model_paths.py` - yolo模型路径解析与模型列表(支持 1..N 编号选择)
- `postprocess.py` - 解码yolo输出，并处理NMS，IoU，返回裁剪前的坐标
- `preprocess.py` - 对图片进行裁剪和归一化的处理，规则取决于训模型时的参数
- `__init__.py`

### backend/pure_color_classify

#### backend/pure_color_classify 根目录文件

- `classify.py` - 纯色分类,按 RGB 规则(CLASSIFY_PIXEL_RULES)对 box 中间区域取色判断
- `__init__.py`

### backend/model_classify

#### backend/model_classify 根目录文件

- `classify.py` - tflite 模型分类(65x65 输入,5 类输出,输入 0-255 原始像素,模型自带 Rescaling)
- `model_paths.py` - 分类模型路径解析与模型列表(支持 models/classify/1.tflite~N.tflite 与子目录)
- `__init__.py`

---

## frontend

- `index.html`
- `main.js`
- `style.css`
  - 前端三件套  

---

## data

### data/Annotations

- 放voc xml  

### data/JPEGImages

- 放.jpg图片

### data/Rejected

- 废弃图片与对应标注(由 reject_manager 移入)
  - `JPEGImages/`
  - `Annotations/`

---

## docs

- `catalog.md`  
  - 本文件,目录说明
- `README.md`
- `rubbish/`  
  - 废弃文档(不再维护)
  - `ARCHITECTURE.md`
  - `bug记录.txt`
  - `learn_markdown.md`
  - `初步架构.txt`
  - `标注工具可行性.txt`
- `screenshots/`  
  - 界面截图

---

## experiments

### experiments/debug

- `debug_yolo_output.py`
- `inspect_tflite.py`
  - 测试yolo模型的输出情况

### experiments/grounding_dino

- `test_one_image.py`
  - 用单张图片调用grounding_dino

### experiments/large_model_legacy

- `batch_runner.py`  
  - 将批量json，转化为voc xml,项目中目前没有使用
- `response_parser.py`  
  - 读取批量json
- `schema.py`  
  - large_model 返回的JOSN协议
- `__init__.py`  
  - 由于使用生成式在测试时使用效果一般，这一部分基本不使用  
  - 于开发初期准备接入多模态大预言模型，因考虑不周，现弃用

### experiments 根目录

- `start_grounding_dino.ps1`  
  - 单独启动 Grounding DINO worker 的辅助脚本(检查路径后使用)

---

## models

### models/classify

#### models/classify 根目录

- `1.tflite` ~ `5.tflite`  
  - 分类模型,按编号命名,前端"模型分类"下拉可切换
  - 输入 65x65x3,输出 5 类(ball/redbag/bluebag/brownbear/whitebear),输入 0-255
- `2026_06_23_13_23_45/`  
  - 分类模型训练产物
  - `model_01_1.0000.h5` - 训练好的 Keras 权重
  - `hist.csv` - 训练历史(loss/accuracy)
  - `acc.png`、`loss.png` - 训练曲线图

### models/large_model

#### models/large_model/responses

- `large_model_result.json`
  - 规范了多模态模型的返回json格式，至26年5月，实测精度不足，弃用

### models/yolov3_mbv2

#### models/yolov3_mbv2/1 ~ 7

- 每个编号目录存放一个模型变体
  - `yolo3_iou_smartcar_final.tflite` - 检测模型
  - `yolo3_anchors.txt` - 对应 anchors
- 说明:
  - `1/` PC 端模型
  - `2/` OpenMV 部署模型,另有 `train_data.txt`
  - `6/`、`7/` 额外有 `yolo3_iou_smartcar_final_with_post_processing.tflite`(带后处理算子版本)
