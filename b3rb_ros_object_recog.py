# Copyright 2025 NXP

# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node
from synapse_msgs.msg import WarehouseShelf

import cv2
import numpy as np

from sensor_msgs.msg import CompressedImage

import pkg_resources

import cv2
import numpy as np
import torch
import torchvision
import time
import yaml
import tflite_runtime.interpreter as tflite

QOS_PROFILE_DEFAULT = 10

PACKAGE_NAME = 'b3rb_ros_aim_india'

RED_COLOR = (0, 0, 255)
BLUE_COLOR = (255, 0, 0)
GREEN_COLOR = (0, 255, 0)


def xywh2xyxy(x):
	""" Converts bounding box from xywh to xyxy format. """
	y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
	y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
	y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
	y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
	y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y

	return y


def non_max_suppression(
	prediction,
	conf_thres=0.25,
	iou_thres=0.45,
	classes=None,
	agnostic=False,
	multi_label=False,
	labels=(),
	max_det=300,
	nm=0,  # number of masks
):
	"""
	Non-Maximum Suppression (NMS) on inference results to reject overlapping detections.

	Returns:
		 list of detections, on (n,6) tensor per image [xyxy, conf, cls]
	"""

	# Checks
	assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
	assert 0 <= iou_thres <= 1, f"Invalid IoU threshold value {iou_thres}, valid values are between 0.0 and 1.0"
	if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
		prediction = prediction[0]  # select only inference output

	device = prediction.device
	mps = "mps" in device.type  # Apple MPS
	if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
		prediction = prediction.cpu()
	bs = prediction.shape[0]  # batch size
	nc = prediction.shape[2] - nm - 5  # number of classes
	xc = prediction[..., 4] > conf_thres  # candidates

	# Settings
	max_wh = 7680  # (pixels) maximum box width and height
	max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
	time_limit = 0.5 + 0.05 * bs  # seconds to quit after
	redundant = True  # require redundant detections
	multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
	merge = False  # use merge-NMS

	t = time.time()
	mi = 5 + nc  # mask start index
	output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
	for xi, x in enumerate(prediction):  # image index, image inference
		# Apply constraints
		x = x[xc[xi]]  # confidence

		# Cat apriori labels if autolabelling
		if labels and len(labels[xi]):
			lb = labels[xi]
			v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
			v[:, :4] = lb[:, 1:5]  # box
			v[:, 4] = 1.0  # conf
			v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
			x = torch.cat((x, v), 0)

		# If none remain process next image
		if not x.shape[0]:
			continue

		# Compute conf
		x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

		# Box/Mask
		box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
		mask = x[:, mi:]  # zero columns if no masks

		# Detections matrix nx6 (xyxy, conf, cls)
		if multi_label:
			i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
			x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
		else:  # best class only
			conf, j = x[:, 5:mi].max(1, keepdim=True)
			x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

		# Filter by class
		if classes is not None:
			x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

		# Check shape
		n = x.shape[0]  # number of boxes
		if not n:  # no boxes
			continue
		x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

		# Batched NMS
		c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
		boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
		i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
		i = i[:max_det]  # limit detections
		if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
			# update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
			iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
			weights = iou * scores[None]  # box weights
			x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
			if redundant:
				i = i[iou.sum(1) > 1]  # require redundancy

		output[xi] = x[i]
		if mps:
			output[xi] = output[xi].to(device)
		if (time.time() - t) > time_limit:
			break  # time limit exceeded

	return output


class ObjectRecognizer(Node):
	""" Initializes object recognizer node with the required publishers and subscriptions.

		Returns:
			None
	"""
	def __init__(self):
		super().__init__('object_recognizer')

		# Subscription for camera images.
		self.subscription_camera = self.create_subscription(
			CompressedImage,
			'/camera/image_raw/compressed',
			self.camera_image_callback,
			QOS_PROFILE_DEFAULT)

		# Publisher for Shelf Objects.
		self.publisher_shelf_objects = self.create_publisher(
			WarehouseShelf,
			'/shelf_objects',
			QOS_PROFILE_DEFAULT)

		# Publisher for output image (for debug purposes).
		self.publisher_object_recog = self.create_publisher(
			CompressedImage,
			"/debug_images/object_recog",
			QOS_PROFILE_DEFAULT)

		resource_name_coco = "../../../../share/ament_index/resource_index/coco.yaml"
		resource_path_coco = pkg_resources.resource_filename(PACKAGE_NAME, resource_name_coco)
		resource_name_yolo = "../../../../share/ament_index/resource_index/yolov5n-int8.tflite"
		resource_path_yolo = pkg_resources.resource_filename(PACKAGE_NAME, resource_name_yolo)

		with open(resource_path_coco) as f:
			self.label_names = yaml.load(f, Loader=yaml.FullLoader)['names']

		ext_delegate_ops = {}
		# Uncomment and replace for running on NavQPlus NPU (On-board neural processing unit).
		# ext_delegate = [tflite.load_delegate("/usr/lib/libvx_delegate.so", ext_delegate_ops)]

		# self.interpreter = tflite.Interpreter(model_path=resource_path_yolo,
		# 				      experimental_delegates=ext_delegate)
		self.interpreter = tflite.Interpreter(model_path=resource_path_yolo)

		self.interpreter.allocate_tensors()

		self.input_details = self.interpreter.get_input_details()
		self.output_details = self.interpreter.get_output_details()


	""" Publishes images for debugging purposes.

		Args:
			publisher: ROS2 publisher of the type sensor_msgs.msg.CompressedImage.
			image: image given by an n-dimensional numpy array.

		Returns:
			None
	"""
	def publish_debug_image(self, publisher, image):
		if image.size:
			message = CompressedImage()
			_, encoded_data = cv2.imencode('.jpg', image)
			message.format = "jpeg"
			message.data = encoded_data.tobytes()
			publisher.publish(message)


	""" Analyzes the image received from /camera/image_raw/compressed to detect shelf objects.
		Publishes the existence of objects in the image on the /shelf_objects topic.

		Args:
			message: "docs.ros.org/en/melodic/api/sensor_msgs/html/msg/CompressedImage.html"

		Returns:
			None
	"""
	def camera_image_callback(self, message):
		# Convert message to an n-dimensional numpy array representation of image.
		np_arr = np.frombuffer(message.data, np.uint8)
		image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
		height, width, _ = image.shape

		# image pre-processing.
		input_size = self.input_details[0]['shape'][1]
		image = cv2.resize(image, (input_size, input_size))
		image = image.astype(np.float32)
		image = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_BGR2RGB)
		image /= 255
		img = np.expand_dims(image, axis=0)

		shelf_objects_message = WarehouseShelf()
		object_count_dict = {}

		# invoke for inference.
		input = self.input_details[0]
		int8 = input["dtype"] == np.uint8  # is TFLite quantized uint8 model
		if int8:
			scale, zero_point = input["quantization"]
			img = (img / scale + zero_point).astype(np.uint8)  # de-scale
		self.interpreter.set_tensor(input["index"], img)

		startTime = time.time()
		self.interpreter.invoke()
		delta = time.time() - startTime
		print("inference time:", '%.1f' % (delta * 1000), "ms")
		y = []
		for output in self.output_details:
			x = self.interpreter.get_tensor(output["index"])
			if int8:
				scale, zero_point = output["quantization"]
				x = (x.astype(np.float32) - zero_point) * scale  # re-scale
			y.append(x)

		image *= 255
		image = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2BGR)

		# processing output.
		for pred in y:
			w, h = self.input_details[0]["shape"][1:3]
			pred[0][..., :4] *= [w, h, w, h]
			pred = torch.tensor(pred)

			max_det = 1000
			conf_thres = 0.25
			pred = non_max_suppression(pred, conf_thres, False, max_det=max_det)

			for i, det in enumerate(pred):
				if len(det):
					for *xyxy, conf, cls in reversed(det):
						start_point = (int(xyxy[0]), int(xyxy[1]))
						end_point = (int(xyxy[2]), int(xyxy[3]))

						object_name = self.label_names[int(cls)]

						if object_name in object_count_dict:
							object_count_dict[object_name] += 1
						else:
							object_count_dict[object_name] = 1

						cv2.rectangle(image, start_point, end_point, GREEN_COLOR, 2)
						cv2.putText(image, self.label_names[int(cls)] + " " + str(round(float(conf), 2)),
							    start_point, cv2.FONT_HERSHEY_SIMPLEX, 1, GREEN_COLOR, 2, cv2.LINE_AA)

			image = cv2.resize(image, (width, height))
			self.publish_debug_image(self.publisher_object_recog, image)

		for key, value in object_count_dict.items():
			shelf_objects_message.object_name.append(key)
			shelf_objects_message.object_count.append(value)

		self.publisher_shelf_objects.publish(shelf_objects_message)


def main(args=None):
	rclpy.init(args=args)

	object_recognizer = ObjectRecognizer()

	rclpy.spin(object_recognizer)

	# Destroy the node explicitly
	# (optional - otherwise it will be done automatically
	# when the garbage collector destroys the node object)
	object_recognizer.destroy_node()
	rclpy.shutdown()


if __name__ == '__main__':
	main()
