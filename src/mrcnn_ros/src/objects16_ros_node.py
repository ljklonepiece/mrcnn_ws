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
from perception_msgs.srv import FetchImage, MRCNNDetect, MRCNNDetectResponse
from perception_msgs.msg import Object2D
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

DEFAULT_LOG_DIR = 'logs'
HEIGHT = 480
WIDTH = 640

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
    NUM_CLASSES = 1 + 16  # Background + 10 classes of objects

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.50

    '''customized setting'''
    IMAGE_MIN_DIM = 480
    IMAGE_MAX_DIM = 640
    MAX_GT_INSTANCES = 10
    POST_NMS_ROIS_INFERENCE = 100
    DETECTION_MAX_INSTANCES = 10


############################################################
#  object to int mapping
############################################################
map = {'source' : 1,
       'chip' : 2,
       'coffee' : 3,
       'sugar' : 4,
       'soup' : 5,
       'can' : 6,
       'jello' : 7,
       'cracker' : 8,
       'apple' : 9,
       'orange' : 10,
       'banana' : 11,
       'bowl' : 12,
       'wood' : 13,
       'cupo' : 14,
       'cupb' : 15,
       'cupg' : 16}

imap = {1 : 'source',
        2 : 'chip',
        3 : 'coffee',
        4 : 'sugar',
        5 : 'soup',
        6 : 'can',
        7 : 'jello',
        8 : 'cracker',
        9 : 'apple',
        10 : 'orange',
        11 : 'banana',
        12 : 'bowl',
        13 : 'wood',
        14 : 'cupo',
        15 : 'cupb',
        16 : 'cupg'}


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
        #weights_path = 'weights/mrcnn_object_weight.h5'
        #weights_path = 'weights/mrcnn_object_weight_new.h5'
        weights_path = 'weights/mrcnn_object_weight16.h5'

        # Load weights
        print("Loading weights ", weights_path)
        self.model.load_weights(weights_path, by_name=True)
        self.graph = tf.get_default_graph()

        self.VIS = True

        self.img_pub = rospy.Publisher('img_splash', Image, queue_size=10)
        self.bridge = CvBridge()
        self.img_client = rospy.ServiceProxy('fetch_image', FetchImage)
        self.detector_server = rospy.Service('mrcnn_detect', MRCNNDetect, self.detect_object)
        print ('wait for service call ...')
        rospy.spin()


    def detect_object(self, req):

        results_count = {}
        results_obj = {}
        detect_iter = 1
        for iter in range(detect_iter):
            # Convert ros image to cv image
            img_msg = self.img_client(int(1)).color
            cv_img = self.bridge.imgmsg_to_cv2(img_msg)
            #cv_img = cv_img[:,:,::-1]

            # Detect objects
            with self.graph.as_default():
                r = self.model.detect([cv_img], verbose=0)[0]

            _, num_obj = self.filter(r['rois'], r['scores'], r['class_ids'])

            if num_obj not in results_count.keys():
                results_count[num_obj] = 1
            else:
                results_count[num_obj] += 1
            results_obj[num_obj] = r
        
        print (results_count)
        max_v = -100
        max_k = 0
        for k in results_count.keys():
            if results_count[k] > max_v:
                max_v = results_count[k]
                max_k = k

        r = results_obj[max_k]

         
        #if self.SPLASH == True:
        masks, rois, dists = self.color_splash(cv_img, r)
        objects = []
        res = MRCNNDetectResponse()

        #res = MRCNNDetectResponse()
        h, w, num = masks.shape
        print (num)
        for i in range(num): 
            obj = Object2D()

            (ys, xs) = np.nonzero(masks[:,:,i])
            mask = [ys[j] * WIDTH + xs[j] for j in range(0, len(ys), 4)]
            #''' for downsampled point cloud '''
            #scale = 1
            #mask = [int(ys[j]/scale * WIDTH /scale+ xs[j]/scale) for j in range(len(ys))]
            #mask = list(set(mask))
            roi = (rois[i]).tolist()
            prob = (dists[i]).tolist()

            obj.mask = mask
            obj.roi = roi
            obj.prob = prob

            objects.append(obj)

        res.objects = objects

        return res

    def filter(self, rois, scores, classes):
        overlap_roi = []
        pixel_distance = 50

        for i, r1 in enumerate(rois):
            ## remove roi that is too small
            area = (r1[2] - r1[0]) * (r1[3] - r1[1])
            center_i = [(r1[2] + r1[0]) / 2.0, (r1[3] + r1[1]) / 2.0]
            if area < 800:
                print ('bbox is too SMALL')
                overlap_roi.append(i)
                continue
            ## if roi is too far away from the center of image, discard it
            if r1[0] < 100 or r1[0] > 400:
                overlap_roi.append(i)
                continue
            for j, r2 in enumerate(rois):
                remove = False
                if i >= j:
                    continue

                dx = min(r1[3], r2[3]) - max(r1[1], r2[1])
                dy = min(r1[2], r2[2]) - max(r1[0], r2[0])

                if dx < 0 or dy < 0:
                    continue

                rect1 = (r1[2] - r1[0]) * (r1[3] - r1[1])
                rect2 = (r2[2] - r2[0]) * (r2[3] - r2[1])

                center_j = [(r2[2] + r2[0]) / 2.0, (r2[3] + r2[1]) / 2.0]

                dist = abs(center_i[0] - center_j[0]) + abs(center_i[1] - center_j[1])
                ratio = 1.0 * rect1 / rect2

                if dist < pixel_distance and 0.8 < ratio < 1.2:
                    print ('$$$===================')
                    print (classes[i])
                    print (classes[j])
                    print ('probably incorrect overlapping of rois')
                    print ('distance: ', dist) 
                    print ('ratio: ', ratio) 
                    print ('===================')
                    remove = True

                if (not remove) and (not classes[i] == classes[j]):
                    continue
                else:
                    print ('going to filter one of them: ')
                    print (classes[i])
                    print (classes[j])
                #print (r1, r2)
                

                area = 1.0 * dx * dy
                #overlap = (area / rect1 + area / rect2) * 0.5
                overlap = max(area/rect1, area/rect2)

                print ('-----')
                print (overlap)
                print ('-----')

                if overlap > 0.20:
                    ## need to prune one
                    if scores[i] > scores[j]:
                        overlap_roi.append(j)
                    else:
                        overlap_roi.append(i)

        overlap_roi = list(set(overlap_roi))

        num_valid = len(rois) - len(overlap_roi)

        return overlap_roi, num_valid

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
        
        ''' 
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
                #overlap = (area / rect1 + area / rect2) * 0.5
                overlap = min(area/rect1, area/rect2)

                if overlap > 0.8:
                    ## need to prune one
                    if scores[i] > scores[j]:
                        overlap_roi.append(j)
                    else:
                        overlap_roi.append(i)

        overlap_roi = list(set(overlap_roi))

        num_valid = len(rois) - len(overlap_roi)
        '''

        overlap_roi, num_valid = self.filter(rois, scores, types)

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
        mask_return = mask.copy()
        rois = np.asarray(rois_f)
        types = np.asarray(types_f)
        scores = np.asarray(scores_f)
        distribution = np.asarray(dist_f)

        
        gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
        # Copy color pixels from the original color image where mask is set
        if mask.shape[-1] > 0:
            # We're treating all instances as one, so collapse the mask into one layer
            mask = (np.sum(mask, -1, keepdims=True) >= 1)
            splash = np.where(mask, image, gray).astype(np.uint8)
    
            for i, r in enumerate(rois):
                [y1, x1, y2, x2] = r
                #print((y2-y1)*(x2-x1))
                score = scores[i]
                type = imap[types[i]]
                cv2.rectangle(splash, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(splash, type + ': ' + str(score)[:4], (x2-80, y2+15), 0,
                        0.5, (255,0,0), 2)
    
        else:
            splash = gray.astype(np.uint8)
        # publish image
        if self.VIS:
            img_msg = self.bridge.cv2_to_imgmsg(splash, 'rgb8')
            img_msg.step = int(img_msg.step)
            for i in range(10):
                self.img_pub.publish(img_msg)
            #file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
            #skimage.io.imsave(file_name, splash)
            # Save output
            #file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
            #skimage.io.imsave(file_name, splash)
            #print("Saved to ", file_name)


        return mask_return, rois, distribution



if __name__ == '__main__':
    rospy.init_node('mrcnn')
    MRCNN_OBJECT()


