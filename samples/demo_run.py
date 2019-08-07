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
start_frame_idx = sys.argv[2]
rec_id = sys.argv[3]
mask_path = sys.argv[4]

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
#IMAGE_DIR = '/home/dmitriy.khvan/ffmpeg-img/'
IMAGE_DIR = '/mnt/bepro-data/data/%s/img1/' % (rec_id)

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
log_filename = '/home/dmitriy.khvan/mrcnn-gcp/samples/dump/tmp/det_%s_%s.txt' % (date_time, camera_id)

webhook_url = 'https://hooks.slack.com/services/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'
log_file = open(log_filename, 'w')

#results.append({
#    "rois": final_rois,
#    "class_ids": final_class_ids,
#    "scores": final_scores,
#    "masks": final_masks,
#})

print ('Processing recording id: ' + rec_id)
print ('Path to image folder: ' + IMAGE_DIR)

mask = cv2.imread(mask_path)

h = mask.shape[0]
w = mask.shape[1]

for y in range(0, h):
    for x in range(0, w):
        if mask[y,x,0] == 0 and mask[y,x,1] == 0 and mask[y,x,2] == 0:
            continue
        else:
            mask[y,x,0] = 255
            mask[y,x,1] = 255
            mask[y,x,2] = 255

for num, filename in enumerate(sorted(glob.glob(os.path.join(IMAGE_DIR,'*.jpg')),key=os.path.getmtime)):
    start = time.time()    

    image = skimage.io.imread(filename)
    image = cv2.bitwise_and(image, mask)

    results = model.detect([image], verbose=0)
    r = results[0]
    
    class_id = r['class_ids']
    det_score = r['scores']

    is_dump = (num % 250 == 0) 

    dump_path = "/home/dmitriy.khvan/mrcnn-gcp/samples/dump/tmp/dump-%06d.jpg" %(num+int(start_frame_idx))
    N = r['rois'].shape[0]

    if is_dump:
        d_image = skimage.io.imread(filename)

    for i in range(N):
        # if not person class
        if class_id[i] != 1:
            continue

        y1, x1, y2, x2 = r['rois'][i]

        # height threshold
        if (y2-y1) <= 30:
            continue

        log_file.write(str(num+int(start_frame_idx))+","+str(x1)+","+str(y1)+","+str(x2)+","+str(y2)+','+str(det_score[i])+"\n") 

        if is_dump:
            cv2.rectangle(d_image, (x1, y1), (x2, y2), (255,0,0), 2)
    if is_dump:
        cv2.imwrite(dump_path,cv2.cvtColor(d_image, cv2.COLOR_RGB2BGR)) 
        
        #https://stackoverflow.com/questions/19756329/can-i-save-a-text-file-in-python-without-closing-it
        log_file.flush()
        # typically the above line would do. however this is used to ensure that the file is written
        os.fsync(log_file.fileno()) 

        #slack_msg3 = {'text': 'frame: ' + str(num) + ' dumped: ' + dump_path}
        #requests.post(webhook_url, json.dumps(slack_msg3))  

    end = time.time()
    slack_msg1 = {'text': 'processing input: ' + filename + '-' +str(camera_id)}
    slack_msg2 = {'text': 'processing time per frame: ' + str(end-start) + ' s.'}    
    
    if num % 100 == 0:
        requests.post(webhook_url, json.dumps(slack_msg1))        
        requests.post(webhook_url, json.dumps(slack_msg2))  

log_file.close()
slack_msg4 = {'text': 'detection finished at frame : ' + str(num) + ' .Check results!'}
requests.post(webhook_url, json.dumps(slack_msg4))  
        
