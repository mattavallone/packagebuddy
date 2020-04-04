#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from packagebuddy.srv import *
import rospy
from copy import deepcopy

MASK_RCNN_MODEL_PATH = '/home/osboxes/catkin_ws/src/packagebuddy/src/siamese-mask-rcnn/lib/Mask_RCNN/'
PROJECT_PATH = '/home/osboxes/catkin_ws/src/packagebuddy/src/siamese-mask-rcnn/'

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
	CHECKPOINT_DIR = '../checkpoints/'
	NUM_TARGETS = 1
	
class LargeEvalConfig(siamese_config.Config):
	# Set batch size to 1 since we'll be running inference on
	# one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1
	NUM_CLASSES = 1 + 1
	NAME = 'coco'
	EXPERIMENT = 'evaluation'
	CHECKPOINT_DIR = '../checkpoints/'
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
		self.model_size = rospy.get_param('~model_size', default='small') # Options are either 'small' or 'large'

		if self.model_size == 'small':
			config = SmallEvalConfig()
		elif model_size == 'large':
			config = LargeEvalConfig()

		# Training schedule needed for evaluating	
		train_schedule = OrderedDict()
		if self.model_size == 'small':
			train_schedule[1] = {"learning_rate": config.LEARNING_RATE, "layers": "heads"}
			train_schedule[120] = {"learning_rate": config.LEARNING_RATE, "layers": "4+"}
			train_schedule[160] = {"learning_rate": config.LEARNING_RATE/10, "layers": "all"}
		elif self.model_size == 'large':
			train_schedule[1] = {"learning_rate": config.LEARNING_RATE, "layers": "heads"}
			train_schedule[240] = {"learning_rate": config.LEARNING_RATE, "layers": "all"}
			train_schedule[320] = {"learning_rate": config.LEARNING_RATE/10, "layers": "all"} 
		
		# Load checkpoint weights
		if self.model_size == 'small':
			checkpoint = 'checkpoints/small_siamese_mrcnn_0160.h5'
		elif self.model_size == 'large':
			checkpoint = 'checkpoints/large_siamese_mrcnn_coco_full_0320.h5'
			
		# Initialize model
		self.siameseMaskRCNN = siamese_model.SiameseMaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

		self.siameseMaskRCNN.load_checkpoint(checkpoint, training_schedule=train_schedule)


		rospy.loginfo('siameseMaskRCNN detector ready...')

		s = rospy.Service('siameseMaskRCNN_detect', ObjectDetect, self._handle_siameseMaskRCNN_detect, buff_size=10000000)

		s.spin()

	def _handle_siameseMaskRCNN_detect(self, req):
		cv_image = None
		detection_array = Detection2DArray()
		detections = []
		boxes = None
		
		try:
			cv_image = self.bridge.imgmsg_to_cv2(req.image, "bgr8")
		except CvBridgeError as e:
			rospy.logerr(e)
		try:
			results = self.siameseMaskRCNN.detect([[target]], [image], verbose=1)
			# STOPPED HERE, get boxes etc.
		except SystemError:
			pass
		# rospy.loginfo('Found {} boxes'.format(len(boxes)))
		for box in boxes:
			detection = Detection2D()
			results = []
			bbox = BoundingBox2D()
			center = Pose2D()

			detection.header = Header()
			detection.header.stamp = rospy.get_rostime()
			# detection.source_img = deepcopy(req.image)

			labels = box.get_all_labels()
			for i in range(0,len(labels)):
				object_hypothesis = ObjectHypothesisWithPose()
				object_hypothesis.id = i
				object_hypothesis.score = labels[i]
				results.append(object_hypothesis)
			
			detection.results = results

			x, y = box.get_xy_center()
			center.x = x
			center.y = y
			center.theta = 0.0
			bbox.center = center

			size_x, size_y = box.get_xy_extents()
			bbox.size_x = size_x
			bbox.size_y = size_y

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
