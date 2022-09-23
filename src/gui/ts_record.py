#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
import rospy
import cv2
from sensor_msgs.msg import Image
from bearnav2.msg import FloatList, SensorsOutput
import matplotlib.pyplot as plt
import numpy as np
import ros_numpy
from nav_msgs.msg import Odometry


last_odom = None
traveled_dist = 0.0


def callback(msg):
    print("received")
    global last_odom, traveled_dist
    if last_odom is None:
        last_odom = msg
        return None
    dx = last_odom.pose.pose.position.x - msg.pose.pose.position.x
    dy = last_odom.pose.pose.position.y - msg.pose.pose.position.y
    dz = last_odom.pose.pose.position.z - msg.pose.pose.position.z
    # add very slight distance during turning to avoid similar images
    last_odom = msg
    ret = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5
    traveled_dist += ret
    f.write(str(traveled_dist) + "\n")


if __name__ == "__main__":
    f = open("dist_out", "w")
    rospy.init_node("dist_record")
    rospy.Subscriber("/total_station_driver/ts_odom", Odometry, callback)
    print("Saving ready")
    rospy.spin()
    f.close()