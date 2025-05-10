#!/usr/bin/env python

from base_classes import DepthSensorProcessing
import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from pfvtr.msg import LidarProcessed
from pfvtr.srv import FetchFov, FetchFovResponse

"""
This file contains node for porcessing lidar data.
To do this it uses an instance of abstract class DepthSensorProcessing.
The result is published to topic /depth_data
"""

class LidarProcessing(DepthSensorProcessing):
	def __init__(self):
		super().__init__()
		rospy.init_node('lidar_processing')
		rospy.loginfo("Lidar Processing started")
		lidar_topic = rospy.get_param("~depth_sensor_topic")
		self.lidar_sub = rospy.Subscriber(lidar_topic, LaserScan, self.callback)
		self.lidar_pub = rospy.Publisher("/lidar_processed", LidarProcessed, queue_size=1)
		self.depth_sensor_max_range = rospy.get_param("~depth_sensor_max_range")
		self.depth_sensor_min_range = rospy.get_param("~depth_sensor_min_range")
		# self.fov = 360								# fov = field of view of depth sensor - different for each episode, 360 (full; view) is default
		# self.fov_service = rospy.Service("fetch_fov", FetchFov, self.fetch_fov)	
		rospy.spin()

	def callback(self, msg_in) -> None:
		# recieve and process depth sensor data
		self.depth_data_raw = msg_in
		self.process_data()
		# publish processed depth sensor data
		msg_out = LidarProcessed()
		msg_out.ranges = self.depth_data_processed
		msg_out.closest_obstacle = self.get_closest_obstacle()
		self.lidar_pub.publish(msg_out)	

	def process_data(self) -> None:				# cut desired fov and map values to desired range			
		# depth_data_cutout = self.cutout_fov()
		self.depth_data_processed = self.normalize_it(self.depth_data_raw.ranges)	
		 
	def cutout_fov(self) -> list[float]:			# fov (field of view) is angle of the final lidar cuttout
		if self.fov < 0 or self.fov > 360:
			raise ValueError("fov < 0 or fov > 360, field of view of lidar has to be in range 0 to 360 degrees, it is: %d", self.fov)
		if self.fov % 2 == 0:		# make fov odd for symetrical field of view of lidar, with no loss in randomness
			self.fov = max(self.fov - 1, 3)
		temp_ranges = list(self.depth_data_raw.ranges)
		temp_ranges[int((self.fov - 1)/2) + 1 : -int((self.fov - 1)/2)] = [-1] * (len(temp_ranges) - self.fov)		
		temp_ranges = tuple(temp_ranges)	
		return temp_ranges
	
	def normalize_it(self, depth_data_cutout) -> list[float]:
		temp_ranges = list(depth_data_cutout)
		for i in range(len(temp_ranges)):
			if temp_ranges[i] != -1:		# normalize only rays inside fov
				temp_ranges[i] = min(temp_ranges[i], self.depth_sensor_max_range)
				temp_ranges[i] = -temp_ranges[i] / self.depth_sensor_max_range + 1		# TODO magic number
		temp_ranges = tuple(temp_ranges)
		return temp_ranges
	
	def get_closest_obstacle(self) -> float:
		return np.min(self.depth_data_raw.ranges)

	def fetch_fov(self, msg_req):
		self.fov = msg_req.fov
		return FetchFovResponse()
	 
if __name__ == '__main__':
	lidar_class_instance = LidarProcessing()
