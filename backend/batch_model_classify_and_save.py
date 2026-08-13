#此文件用于批量模型分类并保存标注(使用 model_classify 的 tflite 分类模型)
#运行: python batch_model_classify_and_save.py

from classify import classify_class_boxes
from xml_writer import write_voc_xml
from xml_reader import read_voc_xml
from config import IMAGE_DIR, CLASSIFIER_MODEL
from pathlib import Path

first_image_name="00000"
last_image_name="04000"

#使用的分类模型编号,对应 models/classify/<编号>.tflite,按需修改
MODEL_SELECT=5

def save_voc_xml(image_name):
	class_boxes=read_voc_xml(image_name).get("class_boxes")

	image_path=IMAGE_DIR/Path(image_name).with_suffix(".jpg").name

	next_class_boxes=classify_class_boxes(
		image_path,
		class_boxes,
		classifier=CLASSIFIER_MODEL,
		model_select=MODEL_SELECT,
	)

	write_voc_xml(image_name,next_class_boxes)

def batch_save(first_image_name,last_image_name):
	image_names=[]

	for i in range(int(first_image_name),int(last_image_name)+1):
		image_names.append(f"{i:05}.jpg")

	for image_name in image_names:
		try:
			save_voc_xml(image_name)
			print(f"标注{image_name}")
		except (RuntimeError,FileNotFoundError):
			continue


if __name__=="__main__":
	print(f"开始模型批分类")
	batch_save(first_image_name,last_image_name)
	print(f"模型批分类结束")
