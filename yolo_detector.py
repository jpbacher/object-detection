import os
import time
import argparse
import numpy as np
import cv2


arg = argparse.ArgumentParser()
arg.add_argument('-i', '--images', required=True, help='path to images')
arg.add_argument('-d', '--darknet', required=True, help='path to config & weights directory')
arg.add_argument('-c', '--confidence', type=float, default=0.5, help='min probability to filter')
arg.add_argument('-t', '--nmsthresh', type=float, default=0.4, help='threshold for NMS')
args = vars(arg.parse_args())

labels_path = os.path.sep.join([args['darknet'], 'coco_names.txt'])
labels = open(labels_path).read().strip().split('\n')

np.random.seed(24)
colors = np.random.uniform(0, 255, size=(len(labels), 3))

cfg_path = os.path.sep.join([args['darknet'], 'yolov3.cfg'])
weights_path = os.path.sep.join([args['darknet'], 'yolov3.weights'])

print(f'*** Reading in YOLO algorithm (DarkNet team) trained on the COCO dataset')
yolo = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)

layer_names = yolo.getLayerNames()
output_layers = [layer_names[layer[0] - 1] for layer in yolo.getUnconnectedOutLayers()]

img = cv2.imread(args['images'])
height, width, channels = img.shape

blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
yolo.setInput(blob)
start_time = time.time()
out_array = yolo.forward(output_layers)
end_time = time.time()
print(f'*** Inference time of YOLO: {end_time - start_time:0.3f} seconds')

bounding_boxes = []
confidences = []
class_labels = []

for out in out_array:
    for detection in out:
        scores = detection[5:]
        label = np.argmax(scores)
        confidence = scores[label]
        if confidence > args['confidence']:
            x_center = int(detection[0] * width)
            y_center = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(x_center - w / 2)
            y = int(y_center - h / 2)
            bounding_boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_labels.append(label)

indexes = cv2.dnn.NMSBoxes(bounding_boxes, confidences, args['confidence'], args['nmsthresh'])

for b in range(len(bounding_boxes)):
    if b in indexes:
        x, y, w, h = bounding_boxes[b]
        color = colors[b]
        text = f'{labels[class_labels[b]]}: {confidences[b]:0.3f}'
        cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness=6)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 3, color, thickness=3)

cv2.imshow('Image', img)
cv2.waitKey(0)
