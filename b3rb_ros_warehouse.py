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
from rclpy.timer import Timer
from rclpy.action import ActionClient
from rclpy.parameter import Parameter
from rclpy.executors import MultiThreadedExecutor
import math
import time
import numpy as np
import cv2
from typing import Optional, Tuple
import asyncio
import threading

from sensor_msgs.msg import Joy
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import CompressedImage

from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped

from nav_msgs.msg import OccupancyGrid
from nav2_msgs.msg import BehaviorTreeLog
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus

from synapse_msgs.msg import Status
from synapse_msgs.msg import WarehouseShelf

from scipy.ndimage import label, center_of_mass
from scipy.spatial.distance import euclidean
from sklearn.decomposition import PCA
from pyzbar.pyzbar import decode

import tkinter as tk
from tkinter import ttk

from scipy.ndimage import label

import os


QOS_PROFILE_DEFAULT = 10
SERVER_WAIT_TIMEOUT_SEC = 5.0

PROGRESS_TABLE_GUI = True


class WindowProgressTable:
	def __init__(self, root, shelf_count):
		self.root = root
		self.root.title("Shelf Objects & QR Link")
		self.root.attributes("-topmost", True)

		self.row_count = 2
		self.col_count = shelf_count

		self.boxes = []
		for row in range(self.row_count):
			row_boxes = []
			for col in range(self.col_count):
				box = tk.Text(root, width=10, height=3, wrap=tk.WORD, borderwidth=1,
					      relief="solid", font=("Helvetica", 14))
				box.insert(tk.END, "NULL")
				box.grid(row=row, column=col, padx=3, pady=3, sticky="nsew")
				row_boxes.append(box)
			self.boxes.append(row_boxes)

		# Make the grid layout responsive.
		for row in range(self.row_count):
			self.root.grid_rowconfigure(row, weight=1)
		for col in range(self.col_count):
			self.root.grid_columnconfigure(col, weight=1)

	def change_box_color(self, row, col, color):
		if 0 <= row < self.row_count and 0 <= col < self.col_count:
			self.boxes[row][col].config(bg=color)
		else:
			print(f"[ERROR] Attempted to color GUI box at invalid index (row={row}, col={col})")


	def change_box_text(self, row, col, text):
		if 0 <= row < self.row_count and 0 <= col < self.col_count:
			self.boxes[row][col].delete(1.0, tk.END)
			self.boxes[row][col].insert(tk.END, text)
		else:
			print(f"[ERROR] Attempted to access GUI box at invalid index (row={row}, col={col})")


box_app = None
def run_gui(shelf_count):
	global box_app
	root = tk.Tk()
	box_app = WindowProgressTable(root, shelf_count)
	root.mainloop()


class WarehouseExplore(Node):
	""" Initializes warehouse explorer node with the required publishers and subscriptions.

		Returns:
			None
	"""
	def __init__(self):
		super().__init__('warehouse_explore')


		self.action_client = ActionClient(
			self,
			NavigateToPose,
			'/navigate_to_pose')

		self.subscription_pose = self.create_subscription(
			PoseWithCovarianceStamped,
			'/pose',
			self.pose_callback,
			QOS_PROFILE_DEFAULT)

		self.subscription_global_map = self.create_subscription(
			OccupancyGrid,
			'/global_costmap/costmap',
			self.global_map_callback,
			QOS_PROFILE_DEFAULT)

		self.subscription_simple_map = self.create_subscription(
			OccupancyGrid,
			'/map',
			self.simple_map_callback,
			QOS_PROFILE_DEFAULT)

		self.subscription_status = self.create_subscription(
			Status,
			'/cerebri/out/status',
			self.cerebri_status_callback,
			QOS_PROFILE_DEFAULT)

		self.subscription_behavior = self.create_subscription(
			BehaviorTreeLog,
			'/behavior_tree_log',
			self.behavior_tree_log_callback,
			QOS_PROFILE_DEFAULT)

		self.subscription_shelf_objects = self.create_subscription(
			WarehouseShelf,
			'/shelf_objects',
			self.shelf_objects_callback,
			QOS_PROFILE_DEFAULT)

		# Subscription for camera images.
		self.subscription_camera = self.create_subscription(
			CompressedImage,
			'/camera/image_raw/compressed',
			self.camera_image_callback,
			QOS_PROFILE_DEFAULT)

		self.publisher_joy = self.create_publisher(
			Joy,
			'/cerebri/in/joy',
			QOS_PROFILE_DEFAULT)

		# Publisher for output image (for debug purposes).
		self.publisher_qr_decode = self.create_publisher(
			CompressedImage,
			"/debug_images/qr_code",
			QOS_PROFILE_DEFAULT)

		self.publisher_shelf_data = self.create_publisher(
			WarehouseShelf,
			"/shelf_data",
			QOS_PROFILE_DEFAULT)

		self.declare_parameter('shelf_count', 1)
		self.declare_parameter('initial_angle', 0.0)

		self.shelf_count = \
			self.get_parameter('shelf_count').get_parameter_value().integer_value
		self.initial_angle = \
			self.get_parameter('initial_angle').get_parameter_value().double_value

		# --- Robot State ---
		self.armed = False
		self.logger = self.get_logger()

		# --- Robot Pose ---
		self.pose_curr = PoseWithCovarianceStamped()
		self.buggy_pose_x = 0.0
		self.buggy_pose_y = 0.0
		self.buggy_center = (0.0, 0.0)
		self.world_center = (0.0, 0.0)

		# --- Map Data ---
		self.simple_map_curr = None
		self.global_map_curr = None

		# --- Goal Management ---
		self.xy_goal_tolerance = 0.5
		self.goal_completed = True  # No goal is currently in-progress.
		self.goal_handle_curr = None
		self.cancelling_goal = False
		self.recovery_threshold = 10

		# --- Goal Creation ---
		self._frame_id = "map"

		# --- Exploration Parameters ---
		self.max_step_dist_world_meters = 7.0
		self.min_step_dist_world_meters = 4.0
		self.full_map_explored_count = 0

		# --- QR Code Data ---
		self.qr_code_str = "Empty"
		if PROGRESS_TABLE_GUI:
			self.table_row_count = 0
			self.table_col_count = 0

		# --- Shelf Data ---
		self.shelf_objects_curr = WarehouseShelf()
		self.last_qr_action_time = 0
		self.last_qr_data = None  # Always reset QR state on node start
		self.qr_scanning_active = False

		self.qr_action_cooldown = 5.0  # seconds
		self.last_publish_time = 0
		

		self.detected_shelves_world = []  # list of (x, y)
		self.shelf_dedup_radius = 0.5  # meters

		self.map_saved = False

		self.shelf_goal_world_coords = []  # List of (x, y) goal points in front of each shelf

		self.detected_shelves_world = []  # ✅ Class-level

		self.navigating_shelves = False

		self.shelf_goal_world_coords = []  # List of (x, y) goals
		self.shelf_goal_orientations = []  # ✅ New: List of corresponding yaw angles

		self.shelf_approach_points = []  # To store both approach points for each shelf

		  # Or lower number like 8

		self.slam_completed = False

		self.last_qr_action_time = 0
		self.last_qr_data = None  # Always reset QR state on node start
		self.qr_scanning_active = False
		self.navigating_shelves = False
		self.shelf_number = 1



	def pose_callback(self, message):
		"""Callback function to handle pose updates.

		Args:
			message: ROS2 message containing the current pose of the rover.

		Returns:
			None
		"""
		self.pose_curr = message
		self.buggy_pose_x = message.pose.pose.position.x
		self.buggy_pose_y = message.pose.pose.position.y
		self.buggy_center = (self.buggy_pose_x, self.buggy_pose_y)

	def simple_map_callback(self, message):
		"""Callback function to handle simple map updates.

		Args:
			message: ROS2 message containing the simple map data.

		Returns:
			None
		"""
		self.simple_map_curr = message
		map_info = self.simple_map_curr.info
		self.world_center = self.get_world_coord_from_map_coord(
			map_info.width / 2, map_info.height / 2, map_info
		)

		# Save map only once when exploration is almost complete
		if not self.map_saved and self.full_map_explored_count >= 2:
			self.save_map_as_image(self.simple_map_curr)
			#print("Image saved:)")
			self.map_saved = True
		    # Call shelf detection immediately after map save
	def global_map_callback(self, message):
		#print("🧠 global_map_callback function called")
		self.global_map_curr = message
		
		height = message.info.height
		width = message.info.width
		resolution = message.info.resolution
		img = np.zeros((height, width), dtype=np.uint8)
		data = np.array(message.data, dtype=np.int8).reshape((height, width))
		unknown_ratio = np.count_nonzero(data == -1) / (height * width)
		
		if unknown_ratio < 0.002:
			self.slam_completed = True
			self.detect_rectangular_shelves_from_image(img, message)

			if self.shelf_approach_points:
				#print("📌 Starting navigation to shelf approach points...")
				threading.Thread(target=self.navigate_to_shelves).start()
				self.qr_scanning_active = True
			

		# === 3. Exploration progress ===
		if not self.goal_completed:
			return
		
		map_array = np.array(self.global_map_curr.data).reshape((height, width))
		total_cells = height * width
		known_cells = np.count_nonzero(map_array != -1)
		explored_percent = (known_cells / total_cells) * 100
		#print(f"[INFO] Map Exploration: {explored_percent:.2f}% explored")

		# === 4. Frontier-based space exploration ===
		frontiers = self.get_frontiers_for_space_exploration(map_array)
		map_info = self.global_map_curr.info

		if frontiers:
			closest_frontier = None
			min_distance_curr = float('inf')

			for fy, fx in frontiers:
				fx_world, fy_world = self.get_world_coord_from_map_coord(fx, fy, map_info)
				distance = euclidean((fx_world, fy_world), self.buggy_center)

				if (distance < min_distance_curr and
					distance <= self.max_step_dist_world_meters and
					distance >= self.min_step_dist_world_meters):
					min_distance_curr = distance
					closest_frontier = (fy, fx)

			if closest_frontier:
				fy, fx = closest_frontier
				goal = self.create_goal_from_map_coord(fx, fy, map_info)
				self.send_goal_from_world_pose(goal)
				#print("🚀 Sending goal for space exploration.")
				return
			else:
				self.max_step_dist_world_meters += 2.0
				self.min_step_dist_world_meters = max(0.25, self.min_step_dist_world_meters - 1.0)

			self.full_map_explored_count = 0
		else:
			self.full_map_explored_count += 1
			#print(f"[INFO] Nothing found in frontiers; count = {self.full_map_explored_count}")
			if self.full_map_explored_count >= 3 and unknown_ratio <= 0.0015 and not self.slam_completed:
				print("✅ [INFO] SLAM completed. Map is fully explored.")
				self.slam_completed = True
				if self.shelf_approach_points:
					self.navigate_to_shelves()

	def get_frontiers_for_space_exploration(self, map_array):
		"""Identifies frontiers for space exploration.

		Args:
			map_array: 2D numpy array representing the map.

		Returns:
			frontiers: List of tuples representing frontier coordinates.
		"""
		frontiers = []
		for y in range(1, map_array.shape[0] - 1):
			for x in range(1, map_array.shape[1] - 1):
				if map_array[y, x] == -1:  # Unknown space and not visited.
					neighbors_complete = [
						(y, x - 1),
						(y, x + 1),
						(y - 1, x),
						(y + 1, x),
						(y - 1, x - 1),
						(y + 1, x - 1),
						(y - 1, x + 1),
						(y + 1, x + 1)
					]

					near_obstacle = False
					for ny, nx in neighbors_complete:
						if map_array[ny, nx] > 0:  # Obstacles.
							near_obstacle = True
							break
					if near_obstacle:
						continue

					neighbors_cardinal = [
						(y, x - 1),
						(y, x + 1),
						(y - 1, x),
						(y + 1, x),
					]

					for ny, nx in neighbors_cardinal:
						if map_array[ny, nx] == 0:  # Free space.
							frontiers.append((ny, nx))
							break

		return frontiers



	def publish_debug_image(self, publisher, image):
		"""Publishes images for debugging purposes.

		Args:
			publisher: ROS2 publisher of the type sensor_msgs.msg.CompressedImage.
			image: Image given by an n-dimensional numpy array.

		Returns:
			None
		"""
		if image.size:
			message = CompressedImage()
			_, encoded_data = cv2.imencode('.jpg', image)
			message.format = "jpeg"
			message.data = encoded_data.tobytes()
			publisher.publish(message)

	def camera_image_callback(self, message):
		"""Callback function to handle incoming camera images.

		Args:
			message: ROS2 message of the type sensor_msgs.msg.CompressedImage.

		Returns:
			None
		"""
		if not self.slam_completed: # Ignore if SLAM is not completed
			return
		if not self.qr_scanning_active:
			return # Ignore if QR scanning is not active	
		#print("Received camera image message.")
		np_arr = np.frombuffer(message.data, np.uint8)
		image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
		#print(f"Decoded image shape: {image.shape}, dtype: {image.dtype}")
		found_qr = False
		# Scan for QR codes
		for barcode in decode(image):
			found_qr = True
			qr_data = barcode.data.decode('utf-8')
			if self.shelf_number!= int(qr_data[0]):
				return
			
			self.shelf_number += 1
			if(self.shelf_number > self.shelf_count):
				self.shelf_number = 1
			self.last_qr_data = qr_data
			self.last_qr_action_time = time.time()
			#print("QR String:", qr_data)
			coordinates = qr_data[1:4]
			#print("Coordinates:", coordinates)

			pts = barcode.polygon
			if len(pts) == 4:
				cv2.polylines(image, [np.array(pts, dtype=np.int32)], True, (0, 255, 0), 2)

			self.publish_debug_image(self.publisher_qr_decode, image)

			shelf_data_message = WarehouseShelf()
			shelf_data_message.object_name = []
			shelf_data_message.object_count = []
			shelf_data_message.qr_decoded = qr_data
			self.publisher_shelf_data.publish(shelf_data_message)
			print(f"Published QR to /shelf_data: {qr_data}")
		if not found_qr:
			#print("No QR code found in this image.")
			# Save image for debugging
			timestamp = int(time.time() * 1000)
			cv2.imwrite(f"/tmp/no_qr_{timestamp}.jpg", image)
			#print(f"Saved image to /tmp/no_qr_{timestamp}.jpg for inspection.")

	def cerebri_status_callback(self, message):
		"""Callback function to handle cerebri status updates.

		Args:
			message: ROS2 message containing cerebri status.

		Returns:
			None
		"""
		if message.mode == 3 and message.arming == 2:
			self.armed = True
		else:
			# Initialize and arm the CMD_VEL mode.
			msg = Joy()
			msg.buttons = [0, 1, 0, 0, 0, 0, 0, 1]
			msg.axes = [0.0, 0.0, 0.0, 0.0]
			self.publisher_joy.publish(msg)

	def behavior_tree_log_callback(self, message):
		"""Alternative method for checking goal status.

		Args:
			message: ROS2 message containing behavior tree log.

		Returns:
			None
		"""
		for event in message.event_log:
			if (event.node_name == "FollowPath" and
				event.previous_status == "SUCCESS" and
				event.current_status == "IDLE"):
				# self.goal_completed = True
				# self.goal_handle_curr = None
				pass


	def shelf_objects_callback(self, message):
		# Use the incoming message to update the current shelf objects
		if not self.slam_completed:
			return
		self.shelf_objects_curr.object_name = list(message.object_name)
		self.shelf_objects_curr.object_count = list(message.object_count)
		if self.last_qr_data is not None:
			self.shelf_objects_curr.qr_decoded = self.last_qr_data
		else:
			self.shelf_objects_curr.qr_decoded = ""

		self.publisher_shelf_data.publish(self.shelf_objects_curr)
		print(f" objects: {self.shelf_objects_curr.object_name}, counts: {self.shelf_objects_curr.object_count}")

		# --- NEW: Always update the correct column based on shelf index ---
		if PROGRESS_TABLE_GUI and box_app is not None:
			obj_str = ""
			for name, count in zip(self.shelf_objects_curr.object_name, self.shelf_objects_curr.object_count):
				obj_str += f"{name}: {count}\n"
			if (obj_str.strip() or self.shelf_objects_curr.qr_decoded.strip()):
				# Determine column index for this shelf
				# Try to get shelf index from message, fallback to 0
				shelf_idx = getattr(message, "shelf_id", 0)

				if not (0 <= shelf_idx < box_app.col_count):
					#print(f"[WARNING] shelf_id {shelf_idx} is out of GUI column range (max={box_app.col_count - 1}). Skipping GUI update.")
					return


				# Fill objects in row 0, QR in row 1, for the correct column
				box_app.change_box_text(0, shelf_idx, obj_str)
				box_app.change_box_color(0, shelf_idx, "cyan")
				box_app.change_box_text(1, shelf_idx, self.shelf_objects_curr.qr_decoded)
				box_app.change_box_color(1, shelf_idx, "yellow")
			else:
				print("[INFO] No objects or QR to update in GUI table.")


	def rover_move_manual_mode(self, speed, turn):
		"""Operates the rover in manual mode by publishing on /cerebri/in/joy.

		Args:
			speed: The speed of the car in float. Range = [-1.0, +1.0];
				   Direction: forward for positive, reverse for negative.
			turn: Steer value of the car in float. Range = [-1.0, +1.0];
				  Direction: left turn for positive, right turn for negative.

		Returns:
			None
		"""
		msg = Joy()
		msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1]
		msg.axes = [0.0, speed, 0.0, turn]
		self.publisher_joy.publish(msg)



	def cancel_goal_callback(self, future):
		"""
		Callback function executed after a cancellation request is processed.

		Args:
			future (rclpy.Future): The future is the result of the cancellation request.
		"""
		cancel_result = future.result()
		if cancel_result:
			self.logger.info("Goal cancellation successful.")
			self.cancelling_goal = False  # Mark cancellation as completed (success).
			return True
		else:
			self.logger.error("Goal cancellation failed.")
			self.cancelling_goal = False  # Mark cancellation as completed (failed).
			return False

	def cancel_current_goal(self):
		"""Requests cancellation of the currently active navigation goal."""
		if self.goal_handle_curr is not None and not self.cancelling_goal:
			self.cancelling_goal = True  # Mark cancellation in-progress.
			self.logger.info("Requesting cancellation of current goal...")
			cancel_future = self.action_client._cancel_goal_async(self.goal_handle_curr)
			cancel_future.add_done_callback(self.cancel_goal_callback)

	def goal_result_callback(self, future):
		"""
		Callback function executed when the navigation goal reaches a final result.

		Args:
			future (rclpy.Future): The future that is result of the navigation action.
		"""
		status = future.result().status
		# NOTE: Refer https://docs.ros2.org/foxy/api/action_msgs/msg/GoalStatus.html.

		if status == GoalStatus.STATUS_SUCCEEDED:
			self.logger.info("Goal completed successfully!")
		else:
			self.logger.warn(f"Goal failed with status: {status}")

		self.goal_completed = True  # Mark goal as completed.
		self.goal_handle_curr = None  # Clear goal handle.

	def goal_response_callback(self, future):
		"""
		Callback function executed after the goal is sent to the action server.

		Args:
			future (rclpy.Future): The future that is server's response to goal request.
		"""
		goal_handle = future.result()
		if not goal_handle.accepted:
			self.logger.warn('Goal rejected :(')
			self.goal_completed = True  # Mark goal as completed (rejected).
			self.goal_handle_curr = None  # Clear goal handle.
		else:
			self.logger.info('Goal accepted :)')
			self.goal_completed = False  # Mark goal as in progress.
			self.goal_handle_curr = goal_handle  # Store goal handle.

			get_result_future = goal_handle.get_result_async()
			get_result_future.add_done_callback(self.goal_result_callback)

	def goal_feedback_callback(self, msg):
		"""
		Callback function to receive feedback from the navigation action.

		Args:
			msg (nav2_msgs.action.NavigateToPose.Feedback): The feedback message.
		"""
		distance_remaining = msg.feedback.distance_remaining
		number_of_recoveries = msg.feedback.number_of_recoveries
		navigation_time = msg.feedback.navigation_time.sec
		estimated_time_remaining = msg.feedback.estimated_time_remaining.sec

		self.logger.debug(f"Recoveries: {number_of_recoveries}, "
				  f"Navigation time: {navigation_time}s, "
				  f"Distance remaining: {distance_remaining:.2f}, "
				  f"Estimated time remaining: {estimated_time_remaining}s")

		if number_of_recoveries > self.recovery_threshold and not self.cancelling_goal:
			self.logger.warn(f"Cancelling. Recoveries = {number_of_recoveries}.")
			self.cancel_current_goal()  # Unblock by discarding the current goal.

	def send_goal_from_world_pose(self, goal_pose):
		"""
		Sends a navigation goal to the Nav2 action server.

		Args:
			goal_pose (geometry_msgs.msg.PoseStamped): The goal pose in the world frame.

		Returns:
			bool: True if the goal was successfully sent, False otherwise.
		"""
		if not self.goal_completed or self.goal_handle_curr is not None:
			return False

		self.goal_completed = False  # Starting a new goal.

		goal = NavigateToPose.Goal()
		goal.pose = goal_pose

		if not self.action_client.wait_for_server(timeout_sec=SERVER_WAIT_TIMEOUT_SEC):
			self.logger.error('NavigateToPose action server not available!')
			return False

		# Send goal asynchronously (non-blocking).
		goal_future = self.action_client.send_goal_async(goal, self.goal_feedback_callback)
		goal_future.add_done_callback(self.goal_response_callback)

		return True



	def _get_map_conversion_info(self, map_info) -> Optional[Tuple[float, float]]:
		"""Helper function to get map origin and resolution."""
		if map_info:
			origin = map_info.origin
			resolution = map_info.resolution
			return resolution, origin.position.x, origin.position.y
		else:
			return None

	def get_world_coord_from_map_coord(self, map_x: int, map_y: int, map_info) \
					   -> Tuple[float, float]:
		"""Converts map coordinates to world coordinates."""
		if map_info:
			resolution, origin_x, origin_y = self._get_map_conversion_info(map_info)
			world_x = (map_x + 0.5) * resolution + origin_x
			world_y = (map_y + 0.5) * resolution + origin_y
			return (world_x, world_y)
		else:
			return (0.0, 0.0)

	def get_map_coord_from_world_coord(self, world_x: float, world_y: float, map_info) \
					   -> Tuple[int, int]:
		"""Converts world coordinates to map coordinates."""
		if map_info:
			resolution, origin_x, origin_y = self._get_map_conversion_info(map_info)
			map_x = int((world_x - origin_x) / resolution)
			map_y = int((world_y - origin_y) / resolution)
			return (map_x, map_y)
		else:
			return (0, 0)

	def _create_quaternion_from_yaw(self, yaw: float) -> Quaternion:
		"""Helper function to create a Quaternion from a yaw angle."""
		cy = math.cos(yaw * 0.5)
		sy = math.sin(yaw * 0.5)
		q = Quaternion()
		q.x = 0.0
		q.y = 0.0
		q.z = sy
		q.w = cy
		return q

	def create_yaw_from_vector(self, dest_x: float, dest_y: float,
				   source_x: float, source_y: float) -> float:
		"""Calculates the yaw angle from a source to a destination point.
			NOTE: This function is independent of the type of map used.

			Input: World coordinates for destination and source.
			Output: Angle (in radians) with respect to x-axis.
		"""
		delta_x = dest_x - source_x
		delta_y = dest_y - source_y
		yaw = math.atan2(delta_y, delta_x)

		return yaw

	def create_goal_from_world_coord(self, world_x: float, world_y: float,
					 yaw: Optional[float] = None) -> PoseStamped:
		"""Creates a goal PoseStamped from world coordinates.
			NOTE: This function is independent of the type of map used.
		"""
		goal_pose = PoseStamped()
		goal_pose.header.stamp = self.get_clock().now().to_msg()
		goal_pose.header.frame_id = self._frame_id

		goal_pose.pose.position.x = world_x
		goal_pose.pose.position.y = world_y

		if yaw is None and self.pose_curr is not None:
			# Calculate yaw from current position to goal position.
			source_x = self.pose_curr.pose.pose.position.x
			source_y = self.pose_curr.pose.pose.position.y
			yaw = self.create_yaw_from_vector(world_x, world_y, source_x, source_y)
		elif yaw is None:
			yaw = 0.0
		else:  # No processing needed; yaw is supplied by the user.
			pass

		goal_pose.pose.orientation = self._create_quaternion_from_yaw(yaw)

		pose = goal_pose.pose.position
		print(f"Goal created: ({pose.x:.2f}, {pose.y:.2f}, yaw={yaw:.2f})")
		return goal_pose

	def create_goal_from_map_coord(self, map_x: int, map_y: int, map_info,
				       yaw: Optional[float] = None) -> PoseStamped:
		"""Creates a goal PoseStamped from map coordinates."""
		world_x, world_y = self.get_world_coord_from_map_coord(map_x, map_y, map_info)

		return self.create_goal_from_world_coord(world_x, world_y, yaw)
	

	def save_map_as_image(self, occupancy_msg):
		height = occupancy_msg.info.height
		width = occupancy_msg.info.width
		data = np.array(occupancy_msg.data).reshape((height, width))

		# Convert occupancy grid to grayscale image
		img = np.zeros((height, width), dtype=np.uint8)
		img[data == 0] = 255        # Free
		img[data == 100] = 0        # Occupied
		img[data == -1] = 128       # Unknown

		img = np.flipud(img)        # Flip vertically for visualization

		# Save image
		cv2.imwrite("saved_map.png", img)
		self.get_logger().info("📸 Saved map as saved_map.png")

		# Also save metadata
		res = occupancy_msg.info.resolution
		origin = occupancy_msg.info.origin.position
		with open("saved_map.yaml", "w") as f:
			f.write("image: saved_map.png\n")
			f.write(f"resolution: {res}\n")
			f.write(f"origin: [{origin.x}, {origin.y}, 0.0]\n")
			f.write("negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n")

		self.get_logger().info("📝 Saved map metadata as saved_map.yaml")




	def navigate_to_shelves(self):
		if not self.slam_completed:
			#print("SLAM not completed yet, cannot navigate to shelves.")
			return

		if self.navigating_shelves:
			#print("Already navigating to shelves, skipping duplicate call.")
			return

		self.qr_scanning_active = True
		self.navigating_shelves = True

		if not self.shelf_approach_points:
			print("No shelf approach points available.")
			self.navigating_shelves = False
			return

		for i, shelf in enumerate(self.shelf_approach_points):
			#print(f"🔄 Processing Shelf-{i+1}")

			# Navigate to long side
			long_pos = shelf['long_side']['position']
			long_yaw = shelf['long_side']['yaw']
			#print(f"  🚀 Navigating to long side at ({long_pos[0]:.2f}, {long_pos[1]:.2f})")
			goal_pose = self.create_goal_from_world_coord(long_pos[0], long_pos[1], yaw=long_yaw)
			sent = self.send_goal_from_world_pose(goal_pose)

			if sent:
				# Wait until the goal is completed
				while not self.goal_completed and rclpy.ok():
					time.sleep(0.1)
				time.sleep(2.0)
			else:
				#print(f"❌ Failed to send long-side goal to Shelf-{i+1}")
				continue

			# Navigate to short side
			short_pos = shelf['short_side']['position']
			short_yaw = shelf['short_side']['yaw']
			#print(f"  🚀 Navigating to short side at ({short_pos[0]:.2f}, {short_pos[1]:.2f})")
			goal_pose = self.create_goal_from_world_coord(short_pos[0], short_pos[1], yaw=short_yaw)
			sent = self.send_goal_from_world_pose(goal_pose)

			if sent:
				while not self.goal_completed and rclpy.ok():
					time.sleep(0.1)
				time.sleep(2.0)
			#else:
				#print(f"❌ Failed to send short-side goal to Shelf-{i+1}")

		#print("✅ Finished navigating to all shelves.")
		self.navigating_shelves = False

		
	def detect_rectangular_shelves_from_image(self, image,message):
			height = message.info.height
			width = message.info.width
			resolution = message.info.resolution
			data = np.array(message.data, dtype=np.int8).reshape((height, width))
			img = np.zeros((height, width), dtype=np.uint8)
			img[data == 0] = 255        # Free
			img[data == 100] = 0        # Occupied
			img[data == -1] = 205       # Unknown

			cv2.imwrite("slam_map.png", img)

			# === Rectangle Detection ===
			blurred = cv2.GaussianBlur(img, (5, 5), 0)
			thresh = cv2.adaptiveThreshold(
				blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
				cv2.THRESH_BINARY_INV, 21, 5
			)
			kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
			closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

			contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
			print(f"🔍 Total contours found: {len(contours)}")
			
			color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
			offset_distance = 3.2  # meters

			# Clear previous detections
			self.shelf_approach_points = []  # List of dicts containing both approach points
			origin_x = message.info.origin.position.x
			origin_y = message.info.origin.position.y

			for i, cnt in enumerate(contours):
				if cv2.contourArea(cnt) < 200:
					continue

				rect = cv2.minAreaRect(cnt)
				(cx, cy), (w, h), angle = rect

				if w == 0 or h == 0:
					continue

				aspect_ratio = max(w, h) / min(w, h)
				diff = abs(w - h)
				area = w * h

				#print(f"  🧪 RotatedRect[{i}]: center=({int(cx)},{int(cy)}), w={int(w)}, h={int(h)}, AR={aspect_ratio:.2f}, area={int(area)}")

				if area > 300 and area < 1500 and  diff > 3 and 0.3 < aspect_ratio < 4.5:
					box = cv2.boxPoints(rect)
					box = np.int0(box)
					new_center = (int(cx), int(cy))

					# Convert to world coordinates
					world_x = origin_x + (cx * resolution)
					world_y = origin_y + (cy * resolution)
					
					theta_rad = math.radians(angle)
					
					# Calculate approach points for both sides
					approach_points = []
					
					# Longer side approach point (dark blue)
					if w > h:
						dx_long = -math.sin(theta_rad)
						dy_long = math.cos(theta_rad)
					else:
						dx_long = math.cos(theta_rad)
						dy_long = math.sin(theta_rad)
					
					long_x = world_x + dx_long * offset_distance
					long_y = world_y + dy_long * offset_distance
					long_yaw = math.atan2(world_y - long_y, world_x - long_x)
					
					# Shorter side approach point (light blue)
					if w > h:
						dx_short = math.cos(theta_rad)
						dy_short = math.sin(theta_rad)
					else:
						dx_short = -math.sin(theta_rad)
						dy_short = math.cos(theta_rad)
					
					short_x = world_x + dx_short * offset_distance
					short_y = world_y + dy_short * offset_distance
					short_yaw = math.atan2(world_y - short_y, world_x - short_x)
					
					# Store both approach points
					shelf_data = {
						'center': (world_x, world_y),
						'long_side': {
							'position': (long_x, long_y),
							'yaw': long_yaw,
							'color': (255, 0, 0)  # Dark blue
						},
						'short_side': {
							'position': (short_x, short_y),
							'yaw': short_yaw,
							'color': (255, 255, 0)  # Light blue
						}
					}
					self.shelf_approach_points.append(shelf_data)
					
					# Visualization
					cv2.drawContours(color_img, [box], 0, (0, 255, 0), 2)
					cv2.circle(color_img, (int(cx), int(cy)), 4, (0, 0, 255), -1)
					
					# Draw both approach points
					for side in ['long_side', 'short_side']:
						point = shelf_data[side]
						map_x = int((point['position'][0] - origin_x) / resolution)
						map_y = int((point['position'][1] - origin_y) / resolution)
						
						cv2.circle(color_img, (map_x, map_y), 6, point['color'], -1)
						cv2.putText(color_img, f"{point['position'][0]:.1f},{point['position'][1]:.1f}",
								(map_x + 5, map_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 
								0.4, point['color'], 1)

			cv2.imwrite("detected_rectangles_from_map.png", color_img)	

def main(args=None):
    rclpy.init(args=args)

    warehouse_explore = WarehouseExplore()

    if PROGRESS_TABLE_GUI:
        gui_thread = threading.Thread(target=run_gui, args=(warehouse_explore.shelf_count,))
        gui_thread.start()

    rclpy.spin(warehouse_explore)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    warehouse_explore.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
	main()
