import os
import os
import json
import shutil
import random
from glob import glob

# Diretórios
LABELME_DIR = "./labelme_dataset"
YOLO_DIR = "./yolo_dataset"
TRAIN_RATIO = 0.8  # 80% treino, 20% validação

# Criando diretórios
os.makedirs(f"{YOLO_DIR}/images/train", exist_ok=True)
os.makedirs(f"{YOLO_DIR}/images/val", exist_ok=True)
os.makedirs(f"{YOLO_DIR}/labels/train", exist_ok=True)
os.makedirs(f"{YOLO_DIR}/labels/val", exist_ok=True)

# Mapeamento de classes
class_names = []

def convert_labelme_to_yolo(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    img_w, img_h = data["imageWidth"], data["imageHeight"]
    
    label_file = json_path.replace(".json", ".txt").replace("labelme_dataset", "yolo_dataset/labels")
    with open(label_file, "w") as f:
        for shape in data["shapes"]:
            label = shape["label"]
            if label not in class_names:
                class_names.append(label)
            class_id = class_names.index(label)
            points = shape["points"]
            x_min = min(p[0] for p in points) / img_w
            y_min = min(p[1] for p in points) / img_h
            x_max = max(p[0] for p in points) / img_w
            y_max = max(p[1] for p in points) / img_h
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

# Processando os arquivos
json_files = glob(f"{LABELME_DIR}/*.json")
random.shuffle(json_files)

train_split = int(len(json_files) * TRAIN_RATIO)
train_files, val_files = json_files[:train_split], json_files[train_split:]

for json_path in train_files:
    convert_labelme_to_yolo(json_path)
    shutil.copy(json_path.replace(".json", ".jpg"), f"{YOLO_DIR}/images/train/")

for json_path in val_files:
    convert_labelme_to_yolo(json_path)
    shutil.copy(json_path.replace(".json", ".jpg"), f"{YOLO_DIR}/images/val/")

# Criando arquivos de configuração
with open(f"{YOLO_DIR}/obj.names", "w") as f:
    f.write("\n".join(class_names))

with open(f"{YOLO_DIR}/obj.data", "w") as f:
    f.write(f"classes = {len(class_names)}\n")
    f.write(f"train = {YOLO_DIR}/train.txt\n")
    f.write(f"valid = {YOLO_DIR}/val.txt\n")
    f.write("names = obj.names\n")
    f.write("backup = backup/")

# Criando listas de treino e validação
with open(f"{YOLO_DIR}/train.txt", "w") as f:
    f.writelines([f"{YOLO_DIR}/images/train/{os.path.basename(img)}\n" for img in train_files])

with open(f"{YOLO_DIR}/val.txt", "w") as f:
    f.writelines([f"{YOLO_DIR}/images/val/{os.path.basename(img)}\n" for img in val_files])

print("Conversão concluída! Base de dados pronta para o treinamento.")

# Script para treinar o YOLO no Darknet
train_script = """
!git clone https://github.com/AlexeyAB/darknet.git
%cd darknet
!make
!./darknet detector train yolo_dataset/obj.data yolo_dataset/yolov4.cfg yolov4.conv.137 -dont_show
"""
with open("train_yolo.sh", "w") as f:
    f.write(train_script)

# Script para transfer learning no Colab
colab_script = """
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/AlexeyAB/darknet.git
%cd darknet
!make

!./darknet detector train /content/drive/MyDrive/yolo_dataset/obj.data /content/drive/MyDrive/yolo_dataset/yolov4.cfg /content/drive/MyDrive/yolov4.conv.137 -dont_show -map
"""
with open("train_colab.py", "w") as f:
    f.write(colab_script)

print("Scripts para treinamento criados! Você pode rodar no Darknet ou no Google Colab.")
