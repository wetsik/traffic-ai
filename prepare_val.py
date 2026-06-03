import os, shutil, random

imgs = os.listdir('yolo_dataset/train/images')
val = random.sample(imgs, 20)

for f in val:
    shutil.copy(f'yolo_dataset/train/images/{f}', f'yolo_dataset/valid/images/{f}')
    label = f.rsplit('.', 1)[0] + '.txt'
    src_label = f'yolo_dataset/train/labels/{label}'
    if os.path.exists(src_label):
        shutil.copy(src_label, f'yolo_dataset/valid/labels/{label}')

print(f'Скопировано {len(val)} файлов в valid')