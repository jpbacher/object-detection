import os
import argparse

from torchvision import models, transforms as T
import cv2
from PIL import Image


arg = argparse.ArgumentParser()
arg.add_argument('-i', '--images', required=True, help='path to images')
arg.add_argument('-c', '--confidence', default=0.75, help='minimum confidence level')
args = vars(arg.parse_args())

labels_path = os.path.sep.join(['darknet', 'torch_coco_names.txt'])
labels = open(labels_path).read().strip().split('\n')


def get_prediction(img_path, threshold):
    img = Image.open(img_path)
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    prediction = model([img])
    pred_class = [labels[p] for p in list(prediction[0]['labels'].numpy())]
    pred_boxes = [[(p[0], p[1]), (p[2], p[3])] for p in list(
        prediction[0]['boxes'].detach().numpy()
    )]
    pred_score = list(prediction[0]['scores'].detach().numpy())
    pred_thresh = [pred_score.index(ps) for ps in pred_score if ps > threshold][-1]
    pred_boxes = pred_boxes[:pred_thresh+1]
    pred_class = pred_class[:pred_thresh+1]
    return pred_boxes, pred_class


def detect_objects(img_path, threshold):
    boxes, label = get_prediction(img_path, threshold)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for b in range(len(boxes)):
        cv2.rectangle(img, boxes[b][0], boxes[b][1], color=(0, 255, 0), thickness=2)
        cv2.putText(img, label[b], boxes[b][0], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
    cv2.imshow('Image', img)
    cv2.waitKey(0)


print(f'*** Reading in Faster-RCNN Resnet50  pre-trained model')
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True).eval()

image_path = args['images']
threshold = args['confidence']
detect_objects(image_path, threshold)
