from config import IMAGE_DIR,IMAGE_EXTENSIONS

def list_images():
    '''返回数据集里所有的文件名'''

    if not IMAGE_DIR.exists():
        IMAGE_DIR.mkdir(parents=True,exist_ok=True)
        return[]
    
    images=[]

    for file_path in IMAGE_DIR.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(file_path.name)

    images.sort()

    return images