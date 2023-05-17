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
from nav_msgs.msg import Odometry


def callback(msg):
    print("received")
    f.write(str(msg.output) + "\n")


if __name__ == "__main__":
    f = open("recorded_alignment.txt", "w")
    rospy.init_node("dist_record")
    rospy.Subscriber("/pfvtr/repeat/output_align", SensorsOutput, callback)
    print("Saving ready")
    rospy.spin()
    f.close()