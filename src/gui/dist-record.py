#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
import rospy
import cv2
from sensor_msgs.msg import Image
from pfvtr.msg import FloatList, SensorsOutput
import matplotlib.pyplot as plt
import numpy as np
import ros_numpy


def callback(msg):
    m = msg.map
    dist = msg.output
    f.write(str(msg.header.stamp) + "," + str(m) + "," + str(dist) + "\n")


if __name__ == "__main__":
    f = open("dist_out", "w")
    rospy.init_node("dist_record")
    rospy.Subscriber("/pfvtr/repeat/output_dist", SensorsOutput, callback)
    print("Saving ready")
    rospy.spin()
    f.close()