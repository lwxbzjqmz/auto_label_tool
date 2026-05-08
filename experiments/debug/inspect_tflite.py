import os

# 减少 TensorFlow 日志干扰
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import sys
from pathlib import Path

BACKEND_DIR=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(BACKEND_DIR))#改变模块搜索列表

from config import YOLOV3_MBV2_PC_MODEL_PATH


def main():
    model_path = YOLOV3_MBV2_PC_MODEL_PATH

    print("=" * 80)
    print("TFLite 模型检查")
    print("=" * 80)
    print(f"模型路径: {model_path}")
    print(f"模型是否存在: {model_path.exists()}")

    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("\n" + "=" * 80)
    print("输入信息 input_details")
    print("=" * 80)

    for i, item in enumerate(input_details):
        print(f"\n输入 {i}")
        print(f"name: {item.get('name')}")
        print(f"index: {item.get('index')}")
        print(f"shape: {item.get('shape')}")
        print(f"shape_signature: {item.get('shape_signature')}")
        print(f"dtype: {item.get('dtype')}")
        print(f"quantization: {item.get('quantization')}")
        print(f"quantization_parameters: {item.get('quantization_parameters')}")

    print("\n" + "=" * 80)
    print("输出信息 output_details")
    print("=" * 80)

    for i, item in enumerate(output_details):
        print(f"\n输出 {i}")
        print(f"name: {item.get('name')}")
        print(f"index: {item.get('index')}")
        print(f"shape: {item.get('shape')}")
        print(f"shape_signature: {item.get('shape_signature')}")
        print(f"dtype: {item.get('dtype')}")
        print(f"quantization: {item.get('quantization')}")
        print(f"quantization_parameters: {item.get('quantization_parameters')}")

    print("\n" + "=" * 80)
    print("所有 tensor 简要信息")
    print("=" * 80)

    tensor_details = interpreter.get_tensor_details()

    for item in tensor_details:
        name = item.get("name")
        index = item.get("index")
        shape = item.get("shape")
        dtype = item.get("dtype")
        print(f"index={index}, name={name}, shape={shape}, dtype={dtype}")


if __name__ == "__main__":
    main()