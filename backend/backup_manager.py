from pathlib import Path
import zipfile

from config import BASE_DIR,ANNOTATION_DIR


BACKUP_DIR=BASE_DIR/"backups"/"annotations"


def make_annotations_backup():
    '''备份Annotations目录为zip文件'''
    xml_files=list(ANNOTATION_DIR.glob("*.xml"))

    if not xml_files:
        raise FileNotFoundError("Annotations目录下没有XML文件可备份")
    
    xml_files.sort(key=_annotation_sort_key)

    min_name=xml_files[0].stem
    max_name=xml_files[-1].stem

    BACKUP_DIR.mkdir(parents=True,exist_ok=True)

    zip_name=f"{min_name}-{max_name}_Annotations.zip"
    zip_path=BACKUP_DIR/zip_name

    with zipfile.ZipFile(zip_path,"w",zipfile.ZIP_DEFLATED) as zip_file:
        for xml_file in xml_files:
            zip_file.write(
                xml_file,
                arcname=xml_file.name
            )

    return {
        "zip_path":zip_path,
        "zip_name":zip_name,
        "count":len(xml_files),
        "min_name":min_name,
        "max_name":max_name,
    }


def _annotation_sort_key(path):
    '''按照标注文件名排序'''
    stem=path.stem

    if stem.isdigit():
        return (0,int(stem))
    
    return (1,stem)