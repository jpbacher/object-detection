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
    path = os.path.sep.join(['\darknet', labels_path])
    labels = open(path).read().strip().split('\n')
    return labels


def get_colors(labels):
    np.random.seed(24)
    colors = np.random.uniform(0, 225, size=(len(labels), 3))
    return colors


def get_cfg_file(cfg_path):
    path = os.path.sep.join(['\darknet', cfg_path])
    return path


def get_weights(weights_path):
    path = os.path.sep.join(['\darknet', weights_path])
    return path


def load_model(cfg_path, weights_path):
    yolo = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    return yolo


def img_to_byte(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='png')
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr


def make_prediction(img, model, labels, colors):
