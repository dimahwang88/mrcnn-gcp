import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import glob
import time
import cv2

import requests
import json
from datetime import datetime

# Root directory of the project
camera_id = sys.argv[1]
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

#%matplotlib inline
#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = '/home/dmitriy.khvan/ffmpeg-img/'

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    BATCH_SIZE = GPU_COUNT * IMAGES_PER_GPU
    #NUM_CLASSES = 2

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

now = datetime.now()
date_time = now.strftime("%m%d%Y_%H%M%S")
log_filename = '/mnt/bepro-bucket/2019-06-13_id6629/%s/det_%s_%s.txt' % (camera_id, date_time)

#print(log_filename)

webhook_url = 'https://hooks.slack.com/services/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'
log_file = open(log_filename, 'w+')

for num, filename in enumerate(sorted(glob.glob(os.path.join(IMAGE_DIR,'*.jpg')))):
    start = time.time()    
    image = skimage.io.imread(filename)
    results = model.detect([image], verbose=0)
    r = results[0]
    #boxes
    is_dump = (num % 300 == 0)
    dump_path = "/mnt/bepro-bucket/2019-06-13_id6629/%s/img_%05d.png" %(camera_id, num+1) 
    N = r['rois'].shape[0]
    
    for i in range(N):
        y1, x1, y2, x2 = r['rois'][i]
        log_file.write(str(num+1)+","+str(x1)+","+str(y1)+","+str(x2)+","+str(y2)+"\n") 
        
        if is_dump:
            cv2.rectangle(image, (x1, y1), (x2, y2), (255,0,0), 2)
    if is_dump:
        cv2.imwrite(dump_path,cv2.cvtColor(image, cv2.COLOR_RGB2BGR))    
        slack_msg3 = {'text': 'frame: ' + str(num) + ' dumped: ' + dump_path}
        requests.post(webhook_url, json.dumps(slack_msg3))  

    end = time.time()
    slack_msg1 = {'text': 'processing input: ' + filename}
    slack_msg2 = {'text': 'processing time per frame: ' + str(end-start) + ' s.'}    
    if num % 10 == 0:
        requests.post(webhook_url, json.dumps(slack_msg1))        
        requests.post(webhook_url, json.dumps(slack_msg2))  

log_file.close()
slack_msg4 = {'text': 'detection finished at frame : ' + str(num) + ' .Check results!'}
requests.post(webhook_url, json.dumps(slack_msg4))  
        
