import os
import argparse
import requests
from flask import Flask, request, Response, jsonify
import json
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

