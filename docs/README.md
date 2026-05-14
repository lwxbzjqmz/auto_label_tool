# auto_label_tool

## 工具介绍

  本工具，于26年5月初开发，用于南京信息工程大学21届智能车蚂蚁搬家数据集标记，可以实现单张物体的所动画框，用grounding_dino自动画框，用自己训的yolo模型画框，同时一键保存成voc xml格式

<details>
<summary>预览voc xml格式</summary>

```xml
<annotation>
    <folder>detect.class</folder>
    <path>00245.jpg</path>
    <path>W:\openmv-ide\object_file\auto_label_tool\data\JPEGImages\00245.jpg</path>
    <source/>
    <database>Unknown</database>
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
            <xmin>175</xmin>
            <ymin>58</ymin>
            <xmax>187</xmax>
            <ymax>68</ymax>
        </bndbox>
    </object>
    <object>
        <name>object</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>190</xmin>
            <ymin>52</ymin>
            <xmax>200</xmax>
            <ymax>64</ymax>
        </bndbox>
    </object>
    <object>
        <name>object</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>0</xmin>
            <ymin>46</ymin>
            <xmax>319</xmax>
            <ymax>238</ymax>
        </bndbox>
    </object>
    <object>
        <name>object</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>163</xmin>
            <ymin>55</ymin>
            <xmax>173</xmax>
            <ymax>68</ymax>
        </bndbox>
    </object>
    <object>
        <name>object</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>185</xmin>
            <ymin>58</ymin>
            <xmax>191</xmax>
            <ymax>64</ymax>
        </bndbox>
    </object>
    <object>
        <name>object</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>35</xmin>
            <ymin>54</ymin>
            <xmax>45</xmax>
            <ymax>61</ymax>
        </bndbox>
    </object>
    <object>
        <name>object</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>223</xmin>
            <ymin>42</ymin>
            <xmax>233</xmax>
            <ymax>47</ymax>
        </bndbox>
    </object>
</annotation>
```

</details>

<details>
<summary> 预览工具截图 </summary>

![截图](screenshots/屏幕截图%202026-05-14%20161653.png)

</details>  

## 使用说明

---

### 环境配置  

1. **硬件需求**：本项目需要对grounding-dino-base模型进行本地部署.

- 笔者为笔记本独显 RTX 5060 Laptop GPU，模型加载的缓存为6.2G  

2. **anaconda的库管理**

- autolabel

    | 库名称 | 版本 |
    | :---: | :---: |
    | Python | 3.10 |
    | Flask | 3.1.3 |
    | TensorFlow | 2.21.0 |
    | NumPy | 2.2.6 |
    | Pillow | 12.2.0 |
    | OpenCV-Python | 4.13.0.92 |
    | Requests | 2.33.1 |

    ```powershell
       conda create -n autolabel python=3.10 -y
       conda activate autolabel
       pip install flask==3.1.3 tensorflow==2.21.0 numpy==2.2.6 pillow==12.2.0 opencv-python==4.13.0.92 requests==2.33.1 
    ```

- grounding_dino  

    | 库名称 | 版本 |
    | :---: | :---: |
    | Python | 3.10 |
    | PyTorch | 2.11.0+cu128 |
    | TorchVision | 0.26.0+cu128 |
    | TorchAudio | 2.11.0+cu128 |
    | Transformers | 5.7.0 |
    | Accelerate | 1.13.0 |
    | HuggingFace Hub | 1.13.0 |
    | SafeTensors | 0.7.0 |
    | Tokenizers | 0.22.2 |
    | NumPy | 2.2.6 |
    | Pillow | 12.1.1 |
    | OpenCV-Python | 4.13.0.92 |
    | tqdm | 4.67.3 |
    | PyYAML | 6.0.3 |
    | psutil | 7.2.2 |
    | Rich | 15.0.0 |

    ```powershell
    conda create -n grounding_dino python=3.10 -y
    conda activate grounding_dino

    python -m pip install --upgrade pip setuptools wheel

    pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 torchaudio==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128

    pip install transformers==5.7.0 accelerate==1.13.0 huggingface-hub==1.13.0 safetensors==0.7.0 tokenizers==0.22.2
    pip install pillow==12.1.1 numpy==2.2.6 opencv-python==4.13.0.92 tqdm==4.67.3 pyyaml==6.0.3 psutil==7.2.2 rich==15.0.0
    ```  

---

### 项目必要参数修改，在config.py中

1.`GROUNDING_DINO_HF_HOME=r"C:\Users\lwx\.cache\huggingface"`
这里的***lwx***需要修改成自己的用户名  

2.`GROUNDING_DINO_PYTHON_PATH=r"W:\anaconda3\envs\grounding_dino\python.exe"`
这个地址修改成自己grounding_dino虚拟环境中python解释器的位置

---

### 项目运行

1. 本项目基本全部使用autolabel虚拟环境，把autolabel作为指定环境

2. 运行.\backend\app.py ,程序自动启动后端，并加载grounding_dino_base模型至缓存，第一次运行会下载模型至本地，之后启动大概需要10秒

3. 在浏览器访问网址 ==http://127.0.0.1:5000== 实现使用

### 项目目录  

[点击跳转catalog.md](catalog.md)  