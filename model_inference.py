import cv2
import torch
from PIL import Image
import glob
import os
import pandas as pd
# Model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/data1/yolov5/runs/train/exp11/weights/best.pt')  # local model
# model = torch.hub.load('path/to/yolov5', 'custom', path='path/to/best.pt', source='local')  # local repo
op=0
# im = '/data1/datasets/b1_350/images/train_02/tile_29824_29760_30464_30400_98_2.png'
# im2 = '/data1/datasets/b1_350/images/train_02/tile_27264_22080_27904_22720_6_3.png'
# im1 = Image.open(im)  # PIL image
# im3 = Image.open(im2)  # PIL image
# im2 = cv2.imread('bus.jpg')[..., ::-1]  # OpenCV image (BGR to RGB)
# imgs = [im1, im3]
if op==0:
    im = '/data1/datasets/b1svs/crop_640/TCGA-3G-AB0T-01Z-00-DX3.3B75DD32-1BD6-4832-A4F1-E6C38FD54E35/0/tile_27264_19520_27904_20160_2.png'
    model.hide_labels = True
    img_txt = '/data1/datasets/b1/tile_27264_19520_27904_20160_2.txt'
    model.hide_conf = True
    model.classes = None
    model.multi_label = True
    model.conf = 0.25
    results = model(im)

    # with open(img_txt, 'w') as f:
    #     f.writelines(str(results.pandas().xyxy[0]))
    results.show()

    # results.print()
if op==1:
    images_path = '/data1/datasets/b1/images/train'
    images_txt = '/data1/datasets/b1/txt'
    images = glob.glob(os.path.join(images_path,'*.png'))
    images_list = []
    for image in images:
        img = Image.open(image)
        images_list.append(img)
    model.conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    # agnostic = False  # NMS class-agnostic
    # multi_label = False  # NMS multiple labels per box
    # classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
    # max_det = 1000  # maximum number of detections per image
    # amp = False  # Automatic Mixed Precision (AMP) inference
    # Inference
    results = model(images_list)
    for i,image_name in enumerate(images):
        img_txt = os.path.join(images_txt, os.path.splitext(os.path.basename(image_name))[0]+'.txt')
        with open(img_txt, 'w') as f:
            f.writelines(str(results.pandas().xyxy[i]))

    results.save()


if op==3:
    im = '/data1/datasets/b1_350/images/train_02/tile_27904_24640_28544_25280_26_0.png'
    model.hide_labels = True
    img_txt = '/data1/datasets/b1/tile_27264_19520_27904_20160_2.txt'
    model.hide_conf = True
    model.classes = None
    model.multi_label = True
    model.conf = 0.25
    results = model(im)

    # with open(img_txt, 'w') as f:
    #     f.writelines(str(results.pandas().xyxy[0]))
    results.show()
# results.show()
#      xmin    ymin    xmax   ymax  confidence  class    name
# 0  749.50   43.50  1148.0  704.5    0.874023      0  person
# 1  433.50  433.50   517.5  714.5    0.687988     27     tie
# 2  114.75  195.75  1095.0  708.0    0.624512      0  person
# 3  986.00  304.00  1028.0  420.0    0.286865     27     tie