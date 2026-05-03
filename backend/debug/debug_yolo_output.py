import math
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf
import sys

BACKEND_DIR = Path(__file__).resolve().parent.parent#改变模块的搜索列表
sys.path.insert(0, str(BACKEND_DIR))

from config import YOLOV3_MBV2_PC_MODEL_PATH, YOLOV3_MBV2_ANCHORS_PATH, IMAGE_DIR


CROP_X = 40
CROP_Y = 0
CROP_W = 240
CROP_H = 240

MODEL_INPUT_W = 144
MODEL_INPUT_H = 144


def sigmoid(x):
    x = float(np.clip(x, -50, 50))
    return 1.0 / (1.0 + math.exp(-x))


def read_anchors():
    text = YOLOV3_MBV2_ANCHORS_PATH.read_text(encoding="utf-8").strip()
    values = [float(x.strip()) for x in text.replace("\n", ",").split(",") if x.strip()]
    return [(values[i], values[i + 1]) for i in range(0, len(values), 2)]


def prepare_input(image_path, input_detail):
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        crop_img = img.crop((CROP_X, CROP_Y, CROP_X + CROP_W, CROP_Y + CROP_H))
        resized = crop_img.resize((MODEL_INPUT_W, MODEL_INPUT_H), Image.BILINEAR)

    arr = np.asarray(resized, dtype=np.uint8)

    scale, zero_point = input_detail["quantization"]

    if input_detail["dtype"] == np.int8:
        arr_float = arr.astype(np.float32) / 255.0
        arr_q = np.round(arr_float / scale + zero_point)
        arr_q = np.clip(arr_q, -128, 127).astype(np.int8)
        return np.expand_dims(arr_q, axis=0)

    if input_detail["dtype"] == np.uint8:
        arr_float = arr.astype(np.float32) / 255.0
        arr_q = np.round(arr_float / scale + zero_point)
        arr_q = np.clip(arr_q, 0, 255).astype(np.uint8)
        return np.expand_dims(arr_q, axis=0)

    arr_float = arr.astype(np.float32) / 255.0
    return np.expand_dims(arr_float, axis=0).astype(np.float32)


def main():
    # 改成你正在测试的真实图片名
    image_name = "00000.jpg"
    image_path = IMAGE_DIR / image_name

    print("image_path:", image_path)
    print("image exists:", image_path.exists())
    print("model_path:", YOLOV3_MBV2_PC_MODEL_PATH)
    print("model exists:", YOLOV3_MBV2_PC_MODEL_PATH.exists())
    print("anchors_path:", YOLOV3_MBV2_ANCHORS_PATH)
    print("anchors:", read_anchors())

    interpreter = tf.lite.Interpreter(model_path=str(YOLOV3_MBV2_PC_MODEL_PATH))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("\ninput_details:")
    print(input_details)

    print("\noutput_details:")
    print(output_details)

    input_tensor = prepare_input(image_path, input_details[0])

    print("\ninput tensor:")
    print("shape:", input_tensor.shape)
    print("dtype:", input_tensor.dtype)
    print("min:", input_tensor.min())
    print("max:", input_tensor.max())

    interpreter.set_tensor(input_details[0]["index"], input_tensor)
    interpreter.invoke()

    raw_output = interpreter.get_tensor(output_details[0]["index"])

    print("\nraw output:")
    print("shape:", raw_output.shape)
    print("dtype:", raw_output.dtype)
    print("min raw:", raw_output.min())
    print("max raw:", raw_output.max())

    scale, zero_point = output_details[0]["quantization"]
    output = (raw_output.astype(np.float32) - zero_point) * scale
    output = output[0]

    print("\ndequant output:")
    print("shape:", output.shape)
    print("min:", output.min())
    print("max:", output.max())
    print("mean:", output.mean())

    grid_h, grid_w, channels = output.shape
    anchors = read_anchors()
    num_anchors = len(anchors)
    values_per_anchor = channels // num_anchors

    print("\ngrid_h:", grid_h)
    print("grid_w:", grid_w)
    print("channels:", channels)
    print("num_anchors:", num_anchors)
    print("values_per_anchor:", values_per_anchor)

    output = output.reshape((grid_h, grid_w, num_anchors, values_per_anchor))

    candidates = []

    for gy in range(grid_h):
        for gx in range(grid_w):
            for ai in range(num_anchors):
                pred = output[gy, gx, ai]

                tx, ty, tw, th, p4, p5 = pred[:6]

                # 三种可能的分数理解方式都打印出来：
                # 1. p4 和 p5 是 logits，需要 sigmoid
                score_sigmoid_mul = sigmoid(p4) * sigmoid(p5)

                # 2. p4 本身就是 objectness
                score_p4_raw = float(p4)

                # 3. p5 本身就是 class/object 分数
                score_p5_raw = float(p5)

                candidates.append({
                    "gx": gx,
                    "gy": gy,
                    "anchor": ai,
                    "raw": [float(x) for x in pred[:6]],
                    "score_sigmoid_mul": score_sigmoid_mul,
                    "score_p4_raw": score_p4_raw,
                    "score_p5_raw": score_p5_raw,
                })

    print("\nTop 20 by sigmoid(p4)*sigmoid(p5):")
    for item in sorted(candidates, key=lambda x: x["score_sigmoid_mul"], reverse=True)[:20]:
        print(item)

    print("\nTop 20 by p4 raw:")
    for item in sorted(candidates, key=lambda x: x["score_p4_raw"], reverse=True)[:20]:
        print(item)

    print("\nTop 20 by p5 raw:")
    for item in sorted(candidates, key=lambda x: x["score_p5_raw"], reverse=True)[:20]:
        print(item)


if __name__ == "__main__":
    main()