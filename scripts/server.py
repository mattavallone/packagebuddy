#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
warnings.filterwarnings("ignore")

import os
import sys
from collections import OrderedDict
from packagebuddy.srv import *
import rospy
from copy import deepcopy
import numpy as np
import cv2

MASK_RCNN_MODEL_PATH = '/home/osboxes/catkin_ws/src/packagebuddy/src/siamese_mask_rcnn/lib/Mask_RCNN/'
PROJECT_PATH = '/home/osboxes/catkin_ws/src/packagebuddy/src/siamese_mask_rcnn/'
SUPPORT_SET = '/home/osboxes/catkin_ws/src/packagebuddy/images/reference-images/'
QUERY_SET = '/home/osboxes/catkin_ws/src/packagebuddy/images/query-images/'

if(MASK_RCNN_MODEL_PATH not in sys.path):
	sys.path.append(MASK_RCNN_MODEL_PATH)

if(PROJECT_PATH not in sys.path):
	sys.path.append(PROJECT_PATH)

from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize

from lib import utils as siamese_utils
from lib import model as siamese_model
from lib import config as siamese_config

from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D, ObjectHypothesisWithPose
from geometry_msgs.msg import PoseWithCovariance, Pose2D
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Configure the model
class SmallEvalConfig(siamese_config.Config):
	# Set batch size to 1 since we'll be running inference on
	# one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1
	NUM_CLASSES = 1 + 1
	NAME = 'coco'
	EXPERIMENT = 'evaluation'
	CHECKPOINT_DIR = PROJECT_PATH + 'checkpoints/'
	NUM_TARGETS = 1

	# TARGET_MAX_DIM = 192
	# TARGET_MIN_DIM = 150
	# IMAGE_MIN_SCALE = 1
	# IMAGE_MIN_DIM = 480
	# IMAGE_MAX_DIM = 640

class LargeEvalConfig(siamese_config.Config):
	# Set batch size to 1 since we'll be running inference on
	# one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1
	NUM_CLASSES = 1 + 1
	NAME = 'coco'
	EXPERIMENT = 'evaluation'
	CHECKPOINT_DIR = PROJECT_PATH + 'checkpoints/'
	NUM_TARGETS = 1
	
	# Large image sizes
	TARGET_MAX_DIM = 192
	TARGET_MIN_DIM = 150
	IMAGE_MIN_DIM = 800
	IMAGE_MAX_DIM = 1024
	# Large model size
	FPN_CLASSIF_FC_LAYERS_SIZE = 1024
	FPN_FEATUREMAPS = 256
	# Large number of rois at all stages
	RPN_ANCHOR_STRIDE = 1
	RPN_TRAIN_ANCHORS_PER_IMAGE = 256
	POST_NMS_ROIS_TRAINING = 2000
	POST_NMS_ROIS_INFERENCE = 1000
	TRAIN_ROIS_PER_IMAGE = 200
	DETECTION_MAX_INSTANCES = 100
	MAX_GT_INSTANCES = 100


class SiameseMaskRCNNServer(object):
	def __init__(self):
		self.bridge = CvBridge()
		self.model_size = rospy.get_param('~model_size', default='large') # Options are either 'small' or 'large'
		self.categories = ['chair', 'couch', 'desk', 'door', 'elevator', 'person'] # six classes
		self.k = rospy.get_param('~shot', default=5) # k-shot learning

		if self.model_size == 'small':
			self.config = SmallEvalConfig()
		elif self.model_size == 'large':
			self.config = LargeEvalConfig()
		
		self.config.NUM_TARGETS = self.k

		# Training schedule needed for evaluating	
		train_schedule = OrderedDict()
		if self.model_size == 'small':
			train_schedule[1] = {"learning_rate": self.config.LEARNING_RATE, "layers": "heads"}
			train_schedule[120] = {"learning_rate": self.config.LEARNING_RATE, "layers": "4+"}
			train_schedule[160] = {"learning_rate": self.config.LEARNING_RATE/10, "layers": "all"}
		elif self.model_size == 'large':
			train_schedule[1] = {"learning_rate": self.config.LEARNING_RATE, "layers": "heads"}
			train_schedule[240] = {"learning_rate": self.config.LEARNING_RATE, "layers": "all"}
			train_schedule[320] = {"learning_rate": self.config.LEARNING_RATE/10, "layers": "all"} 
		
		# Load checkpoint weights
		if self.model_size == 'small':
			checkpoint = PROJECT_PATH + 'checkpoints/small_siamese_mrcnn_0160.h5'
		elif self.model_size == 'large':
			checkpoint = PROJECT_PATH + 'checkpoints/large_siamese_mrcnn_coco_full_0320.h5'
		

		# Directory to save logs and trained model
		MODEL_DIR = os.path.join(PROJECT_PATH, "logs")

		# Initialize model
		self.siameseMaskRCNN = siamese_model.SiameseMaskRCNN(mode="inference", model_dir=MODEL_DIR, config=self.config)

		self.siameseMaskRCNN.load_checkpoint(checkpoint, training_schedule=train_schedule)


		rospy.loginfo('Siamese Mask R-CNN detector ready...')

		s = rospy.Service('siameseMaskRCNN_detect', objectDetect, self._handle_siameseMaskRCNN_detect, buff_size=10000000)

		s.spin()

	def _handle_siameseMaskRCNN_detect(self, req):
		cv_image = None
		detection_array = Detection2DArray()
		detections = []
		boxes = None
		
		# Read in image
		try:
			cv_image = self.bridge.imgmsg_to_cv2(req.image, "bgr8")
		except CvBridgeError as e:
			rospy.logerr(e)

		# Perform inferencing
		try:
			# category = np.random.choice(self.categories, 1) # choose a class randomly
			category = self.categories[3] # door
			ref_image_list = np.random.choice(os.listdir(SUPPORT_SET + category), self.k) # use random k reference images
			# ref_image_list = os.listdir(SUPPORT_SET + category)[:self.k-1] # use first k reference images
			ref_images = []
			for i, img in enumerate(ref_image_list):
				# load reference images
				ref_images.append(cv2.imread(SUPPORT_SET + category + '/' + img))

				# resize them to correct dimensions for model
				ref_images[i], window, scale, padding, crop = utils.resize_image(ref_images[i], 
																				min_dim=self.config.TARGET_MIN_DIM, 
																				max_dim=self.config.TARGET_MAX_DIM, 
																				min_scale=self.config.IMAGE_MIN_SCALE, 
																				mode="square")
				
			outputs = self.siameseMaskRCNN.detect([ref_images], [cv_image], verbose=1)

		except SystemError:
			pass
		
		rospy.loginfo('Found {} objects'.format(len(outputs)))
		
		# Create output message
		for output in outputs:
			print(output)
			detection = Detection2D()
			results = []
			bbox = BoundingBox2D()
			center = Pose2D()

			detection.header = Header()
			detection.header.stamp = rospy.get_rostime()
			# detection.source_img = deepcopy(req.image)

			scores = output['scores']
			labels = output['class_ids']

			for i in range(0,len(labels)):
				object_hypothesis = ObjectHypothesisWithPose()
				object_hypothesis.id = labels[i]
				object_hypothesis.score = scores[i]
				results.append(object_hypothesis)
			
			detection.results = results

			y1, x1, y2, x2 = output['rois']
			center.x = (x2 + x1) / 2
			center.y = (y2 + y1) / 2
			center.theta = 0.0
			bbox.center = center

			bbox.size_x = abs(x2 - x1)
			bbox.size_y = abs(y2 - y1)

			detection.bbox = bbox

			detections.append(detection)

		detection_array.header = Header()
		detection_array.header.stamp = rospy.get_rostime()
		detection_array.detections = detections

		return SiameseMaskRCNNDetectResponse(detection_array)

if __name__ == '__main__':
	rospy.init_node('siameseMaskRCNN_server')
	
	try:
		ys = SiameseMaskRCNNServer()
	except rospy.ROSInterruptException:
		pass
