import os
import argparse
import requests
from flask import Flask, request, Response, jsonify
import json
import io as StringIO
from io import BytesIO
import io
import base64
import numpy as np
import cv2
from PIL import Image


def get_labels(labels_path):
    path = os.path.sep.join(['/darknet', labels_path])
    label_class = open(path).read().strip().split('\n')
    return label_class


def get_colors(labels):
    np.random.seed(24)
    colors = np.random.uniform(0, 225, size=(len(labels), 3))
    return colors


def get_cfg_file(cfg_path):
    path = os.path.sep.join(['/darknet', cfg_path])
    return path


def get_weights(weights_path):
    path = os.path.sep.join(['/darknet', weights_path])
    return path


def load_model(cfg_path, weights_path):
    yolo = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    return yolo


def img_to_byte(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='png')
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr


def make_prediction(img, model, conf_thresh, nms_thresh, labels, colors):
    height, width, channels = img.shape
    layer_names = model.getLayerNames()
    output_layers = [layer_names[layer[0] - 1] for layer in model.getUnconnectedOutLayers()]
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
    model.setInput(blob)
    out_array = model.forward(output_layers)
    bounding_boxes = []
    confidences = []
    class_labels = []
    for out in out_array:
        for detection in out:
            scores = detection[5:]
            label = np.argmax(scores)
            confidence = scores[label]
            if confidence > conf_thresh:
                x_center = int(detection[0] * width)
                y_center = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(x_center - w / 2)
                y = int(y_center - h / 2)
                bounding_boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_labels.append(label)
    indexes = cv2.dnn.NMSBoxes(bounding_boxes, confidences, conf_thresh, nms_thresh)
    for box in range(len(bounding_boxes)):
        if box in indexes:
            x, y, w, h = bounding_boxes[box]
            color = colors[box]
            text = f'{labels[class_labels[box]]}: {confidences[box]:0.3f}'
            cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness=6)
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 3, color, thickness=3)
    return img


labels = get_labels('/coco_names.txt')
colors = get_colors(labels)
cfg = get_cfg_file('/yolov3.cfg')
weights = get_weights('/yolov3.weights')
network = load_model(cfg, weights)


app = Flask(__name__)

