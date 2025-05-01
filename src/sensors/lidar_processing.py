#!/usr/bin/env python

import rospy
from sensor_msgs.msg import LaserScan
from base_classes import DepthSensorProcessing

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
		self.lidar_subscriber = rospy.Subscriber(lidar_topic, LaserScan, self.callback)
		self.depth_sensor_max_range = rospy.get_param("~depth_sensor_max_range")
		self.depth_sensor_min_range = rospy.get_param("~depth_sensor_min_range")
		self.fov = 101			# field of view - set by simul step. tod: dodelat se zdenkem
		rospy.spin()

	def callback(self, msg) -> None:
		self.variable = msg
		self.process_data()
		print("callback processed data:", self.depth_data_processed)

	def process_data(self) -> None:				# cut desired fov and map values to desired range			
		depth_data_cutout = self.cutout_fov(self.fov)
		self.depth_data_processed = self.normalize_it(depth_data_cutout)	
		 
	def cutout_fov(self, fov) -> list[float]:			# fov (field of view) is angle of the final lidar cuttout
		if fov < 0 or fov > 360:
			raise ValueError("fov < 0 or fov > 360, field of view of lidar has to be in range 0 to 360 degrees")
		if fov % 2 == 0:		# make fov odd for symetrical field of view of lidar, with no loss in randomness
			fov = fov + 1
		temp_ranges = list(self.variable.ranges)
		temp_ranges[int((fov - 1)/2) + 1 : -int((fov - 1)/2)] = [-1] * (len(temp_ranges) - fov)		
		temp_ranges = tuple(temp_ranges)	
		return temp_ranges
	
	def normalize_it(self, depth_data_cutout) -> list[float]:
		temp_ranges = list(depth_data_cutout)
		for i in range(len(temp_ranges)):
			temp_ranges[i] = min(temp_ranges[i], self.depth_sensor_max_range)
		temp_ranges = tuple(temp_ranges)
		return temp_ranges
	
	def get_closest_obstacle(self) -> list[float]:
		temp_list = [self.depth_sensor_max_range + 1 if x == -1 else x for x in self.depth_data_processed]
		idx = argmin(temp_list)
		value = min(temp_list)
		return [value, idx]

if __name__ == '__main__':
	lidar_class_instance = LidarProcessing()
