"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import tensorflow as tf

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn_catkin.config import Config
from mrcnn_catkin import model as modellib, utils


# ros modules
import rospy
from mrcnn_msgs.srv import FetchImage, MRCNNDetect, MRCNNDetectResponse
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

DEFAULT_LOG_DIR = 'logs'

############################################################
#  Configurations
############################################################

class ObjectConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "object"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 10  # Background + 10 classes of objects

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.4

    '''customized setting'''
    IMAGE_MIN_DIM = 480
    IMAGE_MAX_DIM = 640
    MAX_GT_INSTANCES = 10
    POST_NMS_ROIS_INFERENCE = 100
    DETECTION_MAX_INSTANCES = 10


############################################################
#  object to int mapping
############################################################
map = {'apple' : 1,
       'straw' : 2,
       'jello' : 3,
       'coke' : 4,
       'cup' : 5,
       'soup' : 6,
       'book' : 7,
       'sugar' : 8,
       'toy' : 9,
       'source' : 10}

imap = {1 : 'apple',
        2 : 'straw',
        3 : 'jello',
        4 : 'coke',
        5 : 'cup',
        6 : 'soup',
        7 : 'book',
        8 : 'sugar',
        9 : 'toy',
        10 : 'source'}


class InferenceConfig(ObjectConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

class MRCNN_OBJECT(object):
    def __init__(self):
        config = InferenceConfig()
        config.display()

        # Create model
        self.model = modellib.MaskRCNN(mode="inference", config=config,
                model_dir=DEFAULT_LOG_DIR)

        # Select weights file to load
        weights_path = 'weights/mrcnn_object_weight.h5'

        # Load weights
        print("Loading weights ", weights_path)
        self.model.load_weights(weights_path, by_name=True)
        self.graph = tf.get_default_graph()

        self.SPLASH = True
        self.img_pub = rospy.Publisher('img_splash', Image, queue_size=10)
        self.bridge = CvBridge()
        self.img_client = rospy.ServiceProxy('fetch_image', FetchImage)
        self.detector_server = rospy.Service('mrcnn_detect', MRCNNDetect, self.detect_object)
        print ('wait for service call ...')
        rospy.spin()


    def detect_object(self, req):

        # Convert ros image to cv image
        img_msg = self.img_client(int(1)).color
        cv_img = self.bridge.imgmsg_to_cv2(img_msg)

        # Detect objects
        with self.graph.as_default():
            r = self.model.detect([cv_img], verbose=1)[0]
        # Color splash
        #print (r)

        if self.SPLASH == True:
            splash = self.color_splash(cv_img, r)
            #file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
            #skimage.io.imsave(file_name, splash)

            # publish image
            img_msg = self.bridge.cv2_to_imgmsg(splash, 'rgb8')
            img_msg.step = int(img_msg.step)
            for i in range(10):
                self.img_pub.publish(img_msg)

            # Save output
            #file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
            #skimage.io.imsave(file_name, splash)
            #print("Saved to ", file_name)

        return MRCNNDetectResponse(int(1))


    def color_splash(self, image, result):
        """Apply color splash effect.
        image: RGB image [height, width, 3]
        mask: instance segmentation mask [height, width, instance count]
    
        Returns result image.
        """
        import cv2
        mask = result['masks']
        rois = result['rois']
        types = result['class_ids']
        scores = result['scores']
        distribution = result['distribution']
        # Make a grayscale copy of the image. The grayscale copy still
        # has 3 RGB channels, though.
        
        overlap_roi = []

        for i, r1 in enumerate(rois):
            for j, r2 in enumerate(rois):
                if i == j:
                    continue
                dx = min(r1[3], r2[3]) - max(r1[1], r2[1])
                dy = min(r1[2], r2[2]) - max(r1[0], r2[0])

                area = 1.0 * dx * dy
                rect1 = (r1[2] - r1[0]) * (r1[3] - r1[1])
                rect2 = (r2[2] - r2[0]) * (r2[3] - r2[1])
                overlap = (area / rect1 + area / rect2) * 0.5

                if overlap > 0.5:
                    ## need to prune one
                    if scores[i] > scores[j]:
                        overlap_roi.append(j)
                    else:
                        overlap_roi.append(i)

        overlap_roi = list(set(overlap_roi))

        num_valid = len(rois) - len(overlap_roi)
        mask_f = np.ones((480, 640, num_valid) , dtype=bool)
        rois_f = []
        types_f = []
        scores_f = []
        dist_f = []

        count = 0

        for i in range(len(rois)):
            if i not in overlap_roi:
                mask_f[:,:, count] = mask[:,:,i]
                rois_f.append(rois[i])
                types_f.append(types[i])
                scores_f.append(scores[i])
                dist_f.append(distribution[i])
                count += 1

        mask = np.asarray(mask_f, dtype=bool)
        rois = np.asarray(rois_f)
        types = np.asarray(types_f)
        scores = np.asarray(scores_f)
        distribution = np.asarray(dist_f)
        
        print (scores)
        print (distribution)
        for d in distribution:
            print (np.sum(d))
    
        gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
        # Copy color pixels from the original color image where mask is set
        if mask.shape[-1] > 0:
            # We're treating all instances as one, so collapse the mask into one layer
            mask = (np.sum(mask, -1, keepdims=True) >= 1)
            splash = np.where(mask, image, gray).astype(np.uint8)
    
            for i, r in enumerate(rois):
                [y1, x1, y2, x2] = r
                score = scores[i]
                type = imap[types[i]]
                cv2.rectangle(splash, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(splash, type + ': ' + str(score)[:4], (x2-80, y2+15), 0,
                        0.5, (255,0,0), 2)
    
        else:
            splash = gray.astype(np.uint8)
        return splash


if __name__ == '__main__':
    rospy.init_node('mrcnn')
    MRCNN_OBJECT()


